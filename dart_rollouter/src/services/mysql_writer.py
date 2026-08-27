import ray
import mysql.connector
import time
import datetime as dt
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

@ray.remote(max_concurrency=1)  # Allow concurrency, no longer serial blocking
class MySQLWriterActor:
    def __init__(self, mysql_cfg, pool_size: int = 32):
        """
        Changed to short connection mode.
        Solves the problem that idle connections in the connection pool are silently dropped 
        by firewalls in specific network environments, causing Python side to wait infinitely 
        (TCP Zombie Connection).
        """
        self.mysql_cfg = mysql_cfg
        # pool_size parameter is retained but no longer used, only for interface compatibility
        
    def _get_new_conn(self, socket_timeout=20.0):
        """
        Create a new connection each time, and set dual timeouts.
        """
        conn = mysql.connector.connect(
            host=self.mysql_cfg.host,
            port=self.mysql_cfg.port,
            user=self.mysql_cfg.username,
            password=self.mysql_cfg.password,
            database=self.mysql_cfg.database,
            connect_timeout=10,  # 1. Handshake timeout (prevent waiting forever when unable to connect)
            use_pure=True, # Use pure Python implementation to avoid issues with C extension modules
        )
        # 2. Transmission timeout (prevent no response when network breaks after connection)
        # MySQLTCPSocket wrapper may not have a direct settimeout method, need to access underlying socket
        if hasattr(conn, '_socket'):
            try:
                # Try to set timeout directly
                if hasattr(conn._socket, 'settimeout'):
                    conn._socket.settimeout(socket_timeout)
                # If MySQLTCPSocket has _socket attribute (underlying socket), then set it
                elif hasattr(conn._socket, '_socket'):
                    conn._socket._socket.settimeout(socket_timeout)
            except (AttributeError, TypeError) as e:
                # If unable to set timeout, log warning but continue (connect_timeout already set)
                print(f"[MySQL Warning] Could not set socket timeout: {e}")
        return conn
        

    # ---------- Upper layer interface 1: Directly provide meta and chunks list ----------
    def insert_run_and_chunks(self,
                              meta: Dict,
                              chunks: List[Dict]) -> Tuple[int, int]:
        """
        Atomically write to rollout_run and rollout_chunk.
        """
        conn = None
        cur = None
        try:
            conn = self._get_new_conn(socket_timeout=20.0)
            conn.autocommit = False
            cur = conn.cursor()

            # 1) upsert rollout_run
            run_sql = """
            INSERT INTO rollout_run
              (run_id, trajectory_id, task_id, trace_id, split_dir,
               reward, num_chunks, used, model_version, create_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON DUPLICATE KEY UPDATE
               task_id=VALUES(task_id),
               trace_id=VALUES(trace_id),
               split_dir=VALUES(split_dir),
               reward=VALUES(reward),
               num_chunks=VALUES(num_chunks),
               used=VALUES(used),
               model_version=VALUES(model_version)
            """
            
            run_params = (
                meta.get('run_id'),
                meta.get('trajectory_id'),
                meta.get('task_id'),
                meta.get('trace_id'),
                meta.get('split_dir'),
                meta.get('reward', 0.0),
                meta.get('num_chunks', len(chunks)),
                meta.get('used', 0),
                meta.get('model_version', '')
            )

            cur.execute(run_sql, run_params)

            # 2) Batch upsert rollout_chunk
            chunk_sql = """
            INSERT INTO rollout_chunk
              (trajectory_id, chunk_index, json_path)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
              json_path=VALUES(json_path)
            """
            chunk_params = [
                (meta['trajectory_id'], int(c['chunk_index']), c['json_path'])
                for c in chunks
            ]
            if chunk_params:
                cur.executemany(chunk_sql, chunk_params)

            conn.commit()
            return (cur.rowcount, len(chunk_params))
            
        except Exception as e:
            print(f"[MySQL Error] insert_run_and_chunks failed: {e}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            raise
        finally:
            if cur:
                try: cur.close()
                except: pass
            if conn:
                try: conn.close()
                except: pass
            
    # ---------- Upper layer interface 2: Write single Run ----------
    def insert_run(self, meta: Dict) -> int:
        conn = None
        cur = None
        try:
            # 1. Get new connection, set 20s timeout
            conn = self._get_new_conn(socket_timeout=20.0)
            
            conn.autocommit = False
            cur = conn.cursor()

            # 1) upsert rollout_run
            run_sql = """
            INSERT INTO rollout_run
              (run_id, trajectory_id, task_id, trace_id,
               reward, used, model_version, instruction, num_chunks, process_reward, create_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON DUPLICATE KEY UPDATE
               task_id=VALUES(task_id),
               trace_id=VALUES(trace_id),
               reward=VALUES(reward),
               used=VALUES(used),
               model_version=VALUES(model_version),
               instruction=VALUES(instruction),
               num_chunks=VALUES(num_chunks),
               process_reward=VALUES(process_reward)
            """
            
            run_params = (
                meta.get('run_id'),
                meta.get('trajectory_id'),
                meta.get('task_id'),
                meta.get('trace_id'),
                meta.get('reward', 0.0),
                meta.get('used', 0),
                meta.get('model_version', ''),
                meta.get('instruction', ''),
                meta.get('num_chunks', 0),
                meta.get('process_reward', None)
            )

            cur.execute(run_sql, run_params)
            conn.commit()
            return cur.rowcount

        except Exception as e:
            print(f"[MySQL Error] insert_run failed: {e}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            raise
        finally:
            if cur:
                try: cur.close()
                except: pass
            if conn:
                try: conn.close()
                except: pass

    # ---------- Heavy task logic ----------

    def process_and_insert_trainable_group(self, run_id, rollout_n: int = 8):
        conn = None
        try:
            # This task is heavy, give longer timeout (e.g. 60s)
            conn = self._get_new_conn(socket_timeout=60.0)
            
            # 1. Latest two versions
            latest_versions = self._fetch_latest_two_versions(conn, run_id)
            
            # 2. Fetch qualified data
            rows = self._fetch_rollout_data(conn, latest_versions, run_id)
            
            # 3. Group by task_id
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["task_id"]].append(row)

            filtered_groups = {}
            for task_id, items in grouped.items():
                items_sorted = sorted(items, key=lambda x: x["create_at"], reverse=True)[:rollout_n]
                if len(items_sorted) == rollout_n:
                    avg_reward = sum(i["reward"] or 0 for i in items_sorted) / len(items_sorted)
                    if 0 <= avg_reward < 1:
                        filtered_groups[task_id] = (items_sorted, avg_reward)
                        
            # 4. Replace when reward mean is 0
            final_groups = {}
            for task_id, (items_sorted, avg_reward) in filtered_groups.items():
                if avg_reward == 0:
                    best_row = self._fetch_best_reward_row(conn, task_id, run_id)
                    if best_row:
                        items_sorted[-1] = best_row
                        final_groups[task_id] = items_sorted
                else:
                    final_groups[task_id] = items_sorted
                    
            # 5. Sort
            results = []
            for task_id, items_sorted in final_groups.items():
                anchor_time = min(i["create_at"] for i in items_sorted)
                results.append((anchor_time, task_id, items_sorted))

            results_sorted = sorted(results, key=lambda x: (x[0]))
            
            # 6. Clear and write to trainable_group
            self._clear_trainable_group(conn)
            
            insert_rows = [row for _, _, items in results_sorted for row in items]
            self._insert_trainable_group_rows(conn, insert_rows)
            
            conn.commit() # Remember to commit
            return len(insert_rows)

        except Exception as e:
            print(f"[MySQL Error] process_and_insert_trainable_group failed: {e}")
            if conn:
                try: conn.rollback()
                except: pass
            raise
        finally:
            if conn:
                try: conn.close()
                except: pass

    # ---------- Internal Helper Methods (Stateless, only receive conn) ----------
    
    def _fetch_latest_two_versions(self, conn, run_id):
        cur = conn.cursor(dictionary=True)
        cur.execute(f"""
            SELECT model_version
            FROM (
                SELECT model_version, create_at
                FROM rollout_run
                WHERE run_id = '{run_id}'
                ORDER BY create_at DESC
            ) sub
            GROUP BY model_version
            ORDER BY MAX(create_at) DESC
            LIMIT 2
        """)
        versions = [row["model_version"] for row in cur.fetchall()]
        cur.close()
        return versions

    def _fetch_rollout_data(self, conn, versions, run_id):
        if not versions:
            return []
        cur = conn.cursor(dictionary=True)
        placeholders = ",".join(["%s"] * len(versions))
        # Note here to prevent SQL injection, use params
        sql = f"""
            SELECT *
            FROM rollout_run
            WHERE used = 0
            AND model_version IN ({placeholders})
            AND run_id = %s
            ORDER BY create_at DESC
        """
        params = tuple(versions) + (run_id,)
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()
        return rows

    def _fetch_best_reward_row(self, conn, task_id, run_id):
        cur = conn.cursor(dictionary=True)
        cur.execute(f"""
            SELECT *
            FROM (
                SELECT *
                FROM rollout_run
                WHERE task_id = %s
                AND reward = 1
                AND run_id = %s
                ORDER BY create_at DESC
                LIMIT 1
            ) sub
        """, (task_id, run_id))
        row = cur.fetchone()
        cur.close()
        return row

    def _clear_trainable_group(self, conn):
        cur = conn.cursor()
        cur.execute("TRUNCATE TABLE trainable_group")
        cur.close()

    def _insert_trainable_group_rows(self, conn, rows, batch_size: int = 500):
        if not rows:
            return

        cols = ["trajectory_id", "task_id", "reward", "model_version", "create_at", "run_id", "process_reward"]
        placeholders = "(" + ",".join(["%s"] * len(cols)) + ")"
        base_sql = f"""
            INSERT INTO trainable_group ({", ".join(cols)})
            VALUES
        """

        cur = conn.cursor()
        try:
            for i in range(0, len(rows), batch_size):
                chunk = rows[i:i+batch_size]
                values_sql = ",".join([placeholders] * len(chunk))
                sql = base_sql + values_sql
                params = []
                for r in chunk:
                    params.extend([
                        r["trajectory_id"],
                        r["task_id"],
                        r["reward"],
                        r["model_version"],
                        r["create_at"],
                        r.get("run_id"),
                        r.get("process_reward")
                    ])
                cur.execute(sql, params)
        finally:
            cur.close()
