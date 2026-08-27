"""
MySQL Datasets Table Management - Using SQLAlchemy ORM (Fixed Session Binding Issues)
"""

import logging
import re
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

from sqlalchemy import (
    create_engine, Column, String, Integer, BigInteger, Text, func, select, update, text, Enum, TIMESTAMP, case, literal, cast, literal_column, tuple_
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects import mysql
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.mysql import INTEGER as MYSQL_INTEGER

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create base class
Base = declarative_base()

# Configuration
DB_CONFIG = {
    'host': 'localhost',
    'username': 'dart_rollouter',
    'password': 'password',
    'database': 'dart_database',
    'port': 3306,
    'charset': 'utf8mb4'
}


def _serialize(value):
    if isinstance(value, datetime):
        return value.isoformat(sep=' ', timespec='seconds')
    return value

class Checkpoint(Base):
    __tablename__ = "checkpoint"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False, server_default=text("''"))
    version = Column(String(50), nullable=False, index=True)            # v1, v2, ...
    run_id = Column(String(191), nullable=False, server_default=text("''"), index=True)  # Added
    status = Column(String(20), nullable=False, index=True)             # PENDING, ...
    path = Column(String(255), nullable=False)
    source = Column(String(50), nullable=True, index=True)             # train, ...
    operator = Column(String(50), nullable=True)
    remark = Column(String(1024), nullable=True)
    config_yaml = Column(Text, nullable=True)

    created_at = Column(
        mysql.TIMESTAMP, server_default=func.current_timestamp(), nullable=False
    )
    updated_at = Column(
        mysql.TIMESTAMP,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        nullable=False,
    )
    deleted_at = Column(mysql.TIMESTAMP, nullable=True)
    started_at = Column(mysql.TIMESTAMP, nullable=True)
    finished_at = Column(mysql.TIMESTAMP, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        return {c.name: _serialize(getattr(self, c.name)) for c in self.__table__.columns}


class RolloutRun(Base):
    __tablename__ = "rollout_run"

    id = Column(BigInteger, nullable=True, primary_key=True)
    run_id = Column(String(191, collation='utf8mb4_unicode_ci'), index=True)
    trajectory_id = Column(String(191, collation='utf8mb4_unicode_ci'))
    task_id = Column(String(191, collation='utf8mb4_unicode_ci'))
    trace_id = Column(String(191, collation='utf8mb4_unicode_ci'))
    split_dir = Column(String(512, collation='utf8mb4_unicode_ci'))
    reward = Column(mysql.DOUBLE(asdecimal=False))
    num_chunks = Column(Integer)
    used = Column(Integer, nullable=False, server_default="0", index=True)
    model_version = Column(String(191, collation='utf8mb4_unicode_ci'))
    instruction = Column(String(1024, collation='utf8mb4_unicode_ci'))
    create_at = Column(mysql.TIMESTAMP, server_default=func.current_timestamp())

    def to_dict(self) -> Dict[str, Any]:
        return {c.name: _serialize(getattr(self, c.name)) for c in self.__table__.columns}
    
class DatasetUsageEvent(Base):
    __tablename__ = "dataset_usage_events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    trajectory_id = Column(String(64), nullable=False)
    run_id        = Column(String(128), nullable=False)
    model_version = Column(String(512))
    used_delta    = Column(Integer, nullable=False, server_default=text("0"))
    event_type    = Column(Enum("INSERT", "UPDATE", "USE", name="dataset_usage_event_type"), nullable=False)
    created_at    = Column(TIMESTAMP, nullable=False, server_default=text("CURRENT_TIMESTAMP"))


class MySQLRolloutORM:
    """
    Encapsulate engine and sessionmaker into class; DB configuration passed through constructor.
    """

    def __init__(self, config: Dict[str, Any] = DB_CONFIG, create_tables_if_missing: bool = True):
        self.config = config
        self.engine = self._build_engine(config)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)
        if create_tables_if_missing:
            Base.metadata.create_all(self.engine)

    @staticmethod
    def _build_engine(conf: Dict[str, Any]):
        # Use URL.create to safely handle passwords and parameters with special characters
        url = URL.create(
            "mysql+pymysql",
            username=conf["username"],  # Fixed: was "user"
            password=conf["password"],
            host=conf["host"],
            port=conf["port"],
            database=conf["database"],
            query={"charset": conf.get("charset", "utf8mb4")},
        )
        return create_engine(url, pool_pre_ping=True, future=True)

    def close_database(self):
        """Release underlying connection pool (optional)."""
        self.engine.dispose()

    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self.engine is not None

    @contextmanager
    def session_scope(self):
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ---- 1) Query rollout_run by run_id: return list[dict] ---------------------
    def get_rollouts_by_run_id(self, run_id: str) -> List[Dict[str, Any]]:
        with self.session_scope() as s:
            rows = s.execute(select(RolloutRun).where(RolloutRun.run_id == run_id)).scalars().all()
            return [r.to_dict() for r in rows]

    # ---- 2) Increment used by 1 based on run_id and trajectory_id --------------------------------------
    def update_rollout_used(self, run_id: str, trajectory_id: str) -> int:
        with self.session_scope() as s:
            stmt = update(RolloutRun).where(
                RolloutRun.run_id == run_id,
                RolloutRun.trajectory_id == trajectory_id,    
            ).values(
                used=func.coalesce(RolloutRun.used, 0) + 1
            )
            result = s.execute(stmt)
            return result.rowcount or 0

    # ---- 3) Insert checkpoint: source=train, status=PENDING, version auto-increment v1,v2...
    def insert_checkpoint(self, path: str, run_id: str, initial: bool = False) -> Dict[str, Any]:
        source = "train"
        status = "PENDING"
        with self.session_scope() as s:
            if initial: # Insert initial version
                # Check if v0 already exists for the same run_id
                exists_v0 = s.execute(
                    select(Checkpoint.id, Checkpoint.path).where(
                        Checkpoint.run_id == run_id,
                        Checkpoint.version == "v0",
                    ).limit(1)
                ).first()

                if exists_v0:
                    print(f"[WARN] initial checkpoint v0 already exists for run_id={run_id}, path={path}; skip insert.")
                    return None

                # Insert fixed version v0
                cp = Checkpoint(
                    source="initial",
                    status=status,
                    version="v0",
                    run_id=run_id,
                    path=path,
                )
                s.add(cp)
                s.flush()
                print(f"[INFO] inserted initial checkpoint: run_id={run_id}, path={path}")
                return cp.to_dict()
            
            # Calculate version only within the same run_id
            existing_versions = s.execute(
                select(Checkpoint.version).where(
                    Checkpoint.source == source,
                    Checkpoint.run_id == run_id,
                )
            ).all()

            max_n = 0
            for (ver,) in existing_versions:
                if not ver:
                    continue
                m = re.fullmatch(r"v(\d+)", ver.strip())
                if m:
                    max_n = max(max_n, int(m.group(1)))
            next_version = f"v{max_n + 1}"

            cp = Checkpoint(
                source=source,
                status=status,
                version=next_version,
                run_id=run_id,
                path=path,
            )
            s.add(cp)
            s.flush()  # Get database-generated fields (such as id / timestamps)
            return cp.to_dict()

    # ---- 4) Get latest n checkpoint paths ------------------------------------
    def get_latest_n_checkpoint_paths(self, run_id: str, n: int = 2) -> List[str]:
        with self.session_scope() as s:
            order_key = cast(func.substr(Checkpoint.version, 2), MYSQL_INTEGER(unsigned=True))
            rows = (
                s.query(Checkpoint.path)
                .filter(Checkpoint.run_id == run_id)
                .order_by(order_key.desc())
                .limit(n)
                .all()
            )
            result = [r[0] for r in rows]

            return result
        
    # Delete all rollout_run records for specified run_id, return affected row count
    def delete_datasets_by_run_id(self, run_id: str) -> int:
        with self.session_scope() as s:
            affected = s.query(RolloutRun)\
                .filter(RolloutRun.run_id == run_id)\
                .delete(synchronize_session=False)
            return affected
    
    # Hard delete checkpoint records for specified run_id, return affected row count
    def delete_checkpoint_by_run_id(self, run_id: str) -> int:
        with self.session_scope() as s:
            affected = (
                s.query(Checkpoint)
                .filter(Checkpoint.run_id == run_id)
                .delete(synchronize_session=False)
            )
            return affected

    # Create a rollout_run record; used defaults to 0
    def create_dataset(
        self,
        trajectory_id: str,
        run_id: str,
        task_id: str,
        trace_id: str,
        model_version: str,
        reward: float,
        used: int = 0,
    ) -> dict:
        with self.session_scope() as s:
            row = RolloutRun(
                trajectory_id=trajectory_id,
                run_id=run_id,
                task_id=task_id,
                trace_id=trace_id,
                model_version=model_version,
                reward=reward,
                used=used if used is not None else 0,
                split_dir = "",
                num_chunks = 0,
            )
            s.add(row)
            try:
                s.flush()  # Get database-generated fields (such as create_at)
            except IntegrityError as e:
                # For example, trajectory_id primary key conflict
                s.rollback()
                raise
            return row.to_dict()
        
    def create_or_update_dataset_with_event(
        self,
        trajectory_id: str,
        run_id: str,
        task_id: str,
        trace_id: str,
        model_version: str,
        reward: float,
        used: int = 0,
    ) -> dict:
        """
        If (trajectory_id, run_id) does not exist: insert main table (used defaults to 0 or passed value), event table records INSERT.
        If already exists: only update model_version, and set used to 0, event table records UPDATE.
        """
        with self.session_scope() as s:
            # 1) Check if already exists
            existing = (
                s.query(RolloutRun)
                 .filter_by(trajectory_id=trajectory_id, run_id=run_id)
                 .one_or_none()
            )

            if existing is None:
                # 2a) Insert
                row = RolloutRun(
                    trajectory_id=trajectory_id,
                    run_id=run_id,
                    task_id=task_id,
                    trace_id=trace_id,
                    model_version=model_version,
                    reward=reward,
                    used=used if used is not None else 0,
                    split_dir="",
                    num_chunks=0,
                )
                s.add(row)
                s.flush()
                event_type = "INSERT"
            else:
                # 2b) Update (only update requested fields)
                existing.model_version = model_version
                existing.used = 0
                s.flush()
                row = existing
                event_type = "UPDATE"

            # 3) Write to event table (record this main table operation)
            evt = DatasetUsageEvent(
                trajectory_id=trajectory_id,
                run_id=run_id,
                model_version=model_version,
                used_delta=0,         # This time doesn't change used, only records structural changes
                event_type=event_type # INSERT or UPDATE
            )
            s.add(evt)
            # Note: session_scope should handle commit; no need to commit here

            return row.to_dict()

    def get_nth_newest_model_success(self, run_id: str, n: int):
        """
        Return (avg_nonneg, count_all, distinct_task_cnt, mapping_task_success)
        - avg_nonneg        : Average of max(reward,0) for all matching rollouts
        - count_all         : Total number of matching rollouts
        - distinct_task_cnt : Number of distinct task_ids after deduplication
        - mapping_task_success  : {task_id: success_rate}  
                            where success_rate = avg(max(reward,0)) grouped by task_id
        """
        paths = self.get_latest_n_checkpoint_paths(run_id=run_id, n=n)
        if len(paths) < n:
            return 0.0, 0, 0, {}, {}

        nth_model_version = paths[-1]

        with self.session_scope() as s:
            # -------------- Overall statistics --------------
            nonneg_sum, count_all, distinct_task_cnt = s.execute(
                select(
                    func.sum(case((RolloutRun.reward >= 0, RolloutRun.reward), else_=0.0)),
                    func.count(literal(1)),
                    func.count(func.distinct(RolloutRun.task_id)),
                ).where(
                    RolloutRun.run_id == run_id,
                    RolloutRun.model_version == nth_model_version,
                )
            ).one()

            count_all = int(count_all or 0)
            distinct_task_cnt = int(distinct_task_cnt or 0)
            if count_all == 0:
                return 0.0, 0, 0, {}, {}

            avg_nonneg = float(nonneg_sum or 0.0) / count_all

            # -------------- Per-task statistics --------------
            task_rows = s.execute(
                select(
                    RolloutRun.task_id,
                    func.avg(case((RolloutRun.reward >= 0, RolloutRun.reward), else_=0.0)),
                    func.count(RolloutRun.task_id),  # Add count
                ).where(
                    RolloutRun.run_id == run_id,
                    RolloutRun.model_version == nth_model_version,
                ).group_by(RolloutRun.task_id)
            ).all()

            # success rate mapping / trajectory count mapping
            mapping_task_success: Dict[str, int] = {}
            mapping_task_counts: Dict[str, int] = {}
            for task_id, avg_val, count in task_rows:
                task_id_str = str(task_id)
                mapping_task_success[task_id_str] = float(avg_val)
                mapping_task_counts[task_id_str] = int(count)

        return avg_nonneg, count_all, distinct_task_cnt, mapping_task_success, mapping_task_counts

    def get_latest_success_for_each_task(self, run_id: str, mapping_task_dynamic_n):
        """
        Return (avg_nonneg, count_all, distinct_task_cnt,
            mapping_task_success, mapping_task_counts)
        - avg_nonneg        : Average of reward >= 0 for all matching rollouts (exclude reward<0 records)
        - count_all         : Total number of valid records for calculating success rate (exclude reward<0)
        - distinct_task_cnt : Number of distinct task_ids after deduplication
        - mapping_task_success  : {task_id: success_rate} (exclude reward<0)
        - mapping_task_counts   : {task_id: valid rollout_count (exclude reward<0)}

        Selection logic: For each task_id, take the latest mapping_task_dynamic_n[task_id] records 
        ordered by create_at descending, but exclude reward<0 records when calculating success rate.
        """
        from sqlalchemy import select, func, case, union_all
        if not mapping_task_dynamic_n:
            return 0.0, 0, 0, {}, {}

        with self.session_scope() as s:
            # 1. First rank latest records for each task
            ranked = (
                select(
                    RolloutRun.task_id,
                    RolloutRun.reward,
                    func.row_number().over(
                        partition_by=RolloutRun.task_id,
                        order_by=RolloutRun.create_at.desc()
                    ).label("rn")
                )
                .where(RolloutRun.run_id == run_id)
                .subquery("ranked")
            )

            # 2. Fix: Use union_all function instead of method
            task_count_selects = []
            
            for tid, cnt in mapping_task_dynamic_n.items():
                if cnt > 0:
                    task_count_selects.append(
                        select(
                            func.cast(tid, String).label("task_id"),
                            func.cast(cnt, Integer).label("max_rn")
                        )
                    )
            
            if task_count_selects:
                # Use union_all function to merge all select statements
                task_count_union = union_all(*task_count_selects)
                task_count_cte = task_count_union.cte("tc")
            else:
                # If no valid task counts, create an empty CTE
                task_count_cte = (
                    select(
                        func.cast(None, String).label("task_id"),
                        func.cast(-1, Integer).label("max_rn")
                    )
                    .where(1 == 0)
                    .cte("tc")
                )

            # 3. First take latest N records, then exclude reward<0 for aggregation
            stmt = (
                select(
                    ranked.c.task_id,
                    func.avg(
                        case((ranked.c.reward >= 0, ranked.c.reward), else_=None)
                    ).label("avg_reward"),
                    func.count(
                        case((ranked.c.reward >= 0, 1), else_=None)
                    ).label("valid_cnt")
                )
                .join(
                    task_count_cte,
                    func.cast(ranked.c.task_id, String) == task_count_cte.c.task_id
                )
                .where(ranked.c.rn <= task_count_cte.c.max_rn)
                .group_by(ranked.c.task_id)
            )

            rows = s.execute(stmt).all()

            # 4. Assemble return values
            mapping_task_success = {str(tid): 0.0 for tid in mapping_task_dynamic_n}
            mapping_task_counts  = {str(tid): 0    for tid in mapping_task_dynamic_n}

            nonneg_sum = 0.0
            count_all  = 0
            for task_id, avg_r, v_cnt in rows:
                tid = str(task_id)
                if v_cnt:
                    mapping_task_success[tid] = float(avg_r or 0.0)
                    mapping_task_counts[tid]  = int(v_cnt)
                    nonneg_sum += float(avg_r or 0.0) * v_cnt
                    count_all  += v_cnt

            distinct_task_cnt = sum(1 for v in mapping_task_counts.values() if v > 0)
            avg_nonneg = nonneg_sum / count_all if count_all else 0.0

            return avg_nonneg, count_all, distinct_task_cnt, mapping_task_success, mapping_task_counts

def create_database_manager() -> MySQLRolloutORM:
    return MySQLRolloutORM(DB_CONFIG)

def load_task_rollout_dict(jsonl_file_path: str) -> Dict[str, int]:
    """
    Load task_id to dynamic_rollout_n mapping dictionary from JSONL file
    
    Args:
        jsonl_file_path (str): JSONL file path
        
    Returns:
        Dict[str, int]: Dictionary with task_id as key and dynamic_rollout_n as value
    """
    task_rollout_dict = {}
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                data = json.loads(line)
                task_id = data['task_id']
                dynamic_rollout_n = data['dynamic_rollout_n']
                task_rollout_dict[task_id] = dynamic_rollout_n
                
    return task_rollout_dict
        
if __name__ == "__main__":
    mysql_writer = create_database_manager()