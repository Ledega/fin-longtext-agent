"""
AFAC2026 - 金融长文本Agent 配置加载器

从 config/settings.yaml 读取配置，提供便捷的路径解析和配置访问接口。
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

# 自动加载 .env 文件（如果存在）
_env_loaded = False


def load_env_file(env_path: Optional[Path] = None) -> None:
    """加载 .env 文件中的环境变量到 os.environ"""
    global _env_loaded
    if _env_loaded:
        return

    env_path = env_path or DEFAULT_ENV_PATH
    if not env_path.exists():
        return  # .env 不存在，跳过（用户可能已手动 export）

    try:
        from dotenv import load_dotenv
        loaded = load_dotenv(dotenv_path=str(env_path), override=False)
        if loaded:
            print(f"[Config] 已加载环境变量: {env_path}")
        _env_loaded = True
    except ImportError:
        # python-dotenv 未安装，尝试手动解析简单格式
        _load_env_simple(env_path)
        _env_loaded = True


def _load_env_simple(env_path: Path) -> None:
    """简易 .env 解析器（不依赖 python-dotenv）"""
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    print(f"[Config] 已加载环境变量（简易模式）: {env_path}")


# 模块导入时自动加载 .env
load_env_file()


class ConfigLoader:
    """配置加载器，解析 settings.yaml 并提供路径和参数访问。"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._config: dict = self._load()

    def _load(self) -> dict:
        """加载 YAML 配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # ──────────────────────────────────────────────
    # 基础路径
    # ──────────────────────────────────────────────

    @property
    def data_root(self) -> Path:
        """数据集根目录的绝对路径"""
        return PROJECT_ROOT / self._config["data_root"]

    def abs_path(self, relative_path: str) -> Path:
        """将相对路径转为绝对路径（相对于 data_root）"""
        return self.data_root / relative_path

    # ──────────────────────────────────────────────
    # 文档路径
    # ──────────────────────────────────────────────

    def get_domain_raw_dir(self, domain: str) -> Path:
        """获取某个领域的原始文档目录

        Args:
            domain: 领域标识，如 "insurance", "regulatory", "financial_contracts", 
                     "financial_reports", "research"

        Returns:
            该领域原始文档目录的绝对路径
        """
        cfg = self._config["raw_docs"]["domains"][domain]
        return self.data_root / self._config["raw_docs"]["base_dir"] / cfg["dir"]

    def get_domain_doc_paths(self, domain: str) -> List[Path]:
        """获取某个领域的所有文档文件路径列表

        对 regulatory 领域，返回 html 目录下的文件（已解析的 HTML 格式）。
        其他领域返回目录下所有匹配的文件。
        """
        raw_dir = self.get_domain_raw_dir(domain)
        cfg = self._config["raw_docs"]["domains"][domain]

        if domain == "regulatory":
            # 监管法规优先使用 html 格式（已清洗的文本）
            html_dir = raw_dir / cfg.get("html_dir", "html")
            if html_dir.exists():
                return sorted(html_dir.glob("*.html"))
            # fallback 到 txt
            txt_dir = raw_dir / cfg.get("txt_dir", "txt")
            if txt_dir.exists():
                return sorted(txt_dir.glob("*.txt"))
            # 最后 fallback 到 attachments
            att_dir = raw_dir / cfg.get("attachments_dir", "attachments")
            if att_dir.exists():
                return sorted(att_dir.glob("*.pdf"))
            return []

        # 其他领域：按 doc_pattern 匹配
        pattern = cfg.get("doc_pattern", "*")
        return sorted(raw_dir.glob(pattern))

    def get_all_domains(self) -> List[str]:
        """获取所有领域标识列表"""
        return list(self._config["raw_docs"]["domains"].keys())

    def get_domain_info(self, domain: str) -> dict:
        """获取某个领域的描述信息"""
        return self._config["raw_docs"]["domains"].get(domain, {})

    # ──────────────────────────────────────────────
    # 问题文件路径
    # ──────────────────────────────────────────────

    def get_question_files(self, split: str = "A") -> List[Path]:
        """获取指定榜单的问题文件路径列表

        Args:
            split: "A" 或 "B"

        Returns:
            问题 JSON 文件路径列表
        """
        group_key = f"group_{split.lower()}"
        group_cfg = self._config["questions"].get(group_key, {})
        q_dir = group_cfg.get("dir", "")
        files = group_cfg.get("files", [])

        base = self.data_root / self._config["questions"]["base_dir"]
        return [base / q_dir / f for f in files]

    def load_all_questions(self, split: str = "A") -> List[dict]:
        """加载指定榜单的所有问题

        Args:
            split: "A" 或 "B"

        Returns:
            问题字典列表
        """
        import json

        questions = []
        for q_file in self.get_question_files(split):
            if q_file.exists():
                with open(q_file, "r", encoding="utf-8") as f:
                    questions.extend(json.load(f))
        return questions

    def get_domain_questions(self, domain: str, split: str = "A") -> List[dict]:
        """加载特定领域的问题

        Args:
            domain: 领域标识
            split: "A" 或 "B"

        Returns:
            该领域的问题字典列表
        """
        import json

        domain_file_map = {
            "financial_contracts": "financial_contracts_questions.json",
            "financial_reports": "financial_reports_questions.json",
            "insurance": "insurance_questions.json",
            "regulatory": "regulatory_questions.json",
            "research": "research_questions.json",
        }

        filename = domain_file_map.get(domain)
        if not filename:
            return []

        group_key = f"group_{split.lower()}"
        group_cfg = self._config["questions"].get(group_key, {})
        q_dir = group_cfg.get("dir", "")
        base = self.data_root / self._config["questions"]["base_dir"]
        q_file = base / q_dir / filename

        if q_file.exists():
            with open(q_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    # ──────────────────────────────────────────────
    # 模型配置
    # ──────────────────────────────────────────────

    @property
    def model_config(self) -> dict:
        """获取模型配置"""
        return self._config.get("model", {})

    @property
    def default_model(self) -> str:
        """默认模型名称"""
        return self.model_config.get("default_model", "qwen3-plus")

    @property
    def api_key(self) -> Optional[str]:
        """从环境变量获取 API Key"""
        env_var = self.model_config.get("api_key_env", "QWEN_API_KEY")
        return os.environ.get(env_var)

    # ──────────────────────────────────────────────
    # 输出路径
    # ──────────────────────────────────────────────

    @property
    def output_answer_file(self) -> Path:
        """答案 CSV 文件路径"""
        return PROJECT_ROOT / self._config["output"]["answer_file"]

    @property
    def output_evidence_file(self) -> Path:
        """证据 JSON 文件路径"""
        return PROJECT_ROOT / self._config["output"]["evidence_file"]

    # ──────────────────────────────────────────────
    # Token 预算
    # ──────────────────────────────────────────────

    @property
    def token_budget(self) -> int:
        """Token 预算上限"""
        return self._config.get("token_budget", 5000000)

    # ──────────────────────────────────────────────
    # PDF 解析配置
    # ──────────────────────────────────────────────

    @property
    def pdf_parser_config(self) -> dict:
        """PDF 解析器配置"""
        return self._config.get("pdf_parser", {})

    # ──────────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────────

    def find_doc_by_id(self, doc_id: str) -> Optional[Path]:
        """根据 doc_id 在所有领域中查找文档文件

        Args:
            doc_id: 文档 ID，如 "text01", "byd_2024_annual", "strict_csrc_035"

        Returns:
            文档文件路径，未找到则返回 None
        """
        doc_id = doc_id.lower()

        for domain in self.get_all_domains():
            for doc_path in self.get_domain_doc_paths(domain):
                # 匹配文件名（不含扩展名）
                if doc_path.stem.lower() == doc_id:
                    return doc_path
                # 也匹配文件名中包含 doc_id 的情况
                if doc_id in doc_path.stem.lower():
                    return doc_path

        # 特殊处理 regulatory 的 html 文件
        raw_base = self.data_root / self._config["raw_docs"]["base_dir"]
        reg_dir = raw_base / "regulatory"
        html_dir = reg_dir / "html"
        if html_dir.exists():
            for html_file in html_dir.glob("*.html"):
                if html_file.stem.lower() == doc_id:
                    return html_file

        return None

    def get_qid_domain(self, qid: str) -> Optional[str]:
        """根据 qid 推断所属领域

        Args:
            qid: 题目 ID，如 "fc_a_001", "ins_a_002"

        Returns:
            领域标识，如 "financial_contracts", "insurance"
        """
        prefix_map = {
            "fc": "financial_contracts",
            "fr": "financial_reports",
            "ins": "insurance",
            "reg": "regulatory",
            "res": "research",
        }
        prefix = qid.split("_")[0] if "_" in qid else ""
        return prefix_map.get(prefix)

    def __repr__(self) -> str:
        return f"ConfigLoader(data_root={self.data_root})"


# 全局单例
_config_loader: Optional[ConfigLoader] = None


def get_config() -> ConfigLoader:
    """获取全局配置加载器单例"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader