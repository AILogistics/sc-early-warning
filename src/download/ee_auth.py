"""Google Earth Engine 初始化封装。"""
import sys
import ee


def init_ee(project: str = "ee-qinka-sc-warning") -> None:
    """初始化 GEE,project 需在 GEE Console 创建。

    Args:
        project: GEE Cloud project ID。首次使用前需:
            1. 注册 https://code.earthengine.google.com/
            2. 运行 `python -c "import ee; ee.Authenticate()"` 完成 OAuth
            3. 在 https://console.cloud.google.com/earth-engine 创建 project

    Raises:
        RuntimeError: GEE 初始化失败(未认证 / project 不存在 / 网络问题)
    """
    try:
        ee.Initialize(project=project)
        # 验证:获取一个已知影像的元数据
        srtm = ee.Image("USGS/SRTMGL1_003")
        _ = srtm.bandNames().getInfo()
        print(f"[OK] GEE initialized, project={project}", file=sys.stderr)
    except Exception as e:
        raise RuntimeError(
            f"GEE init failed: {e}. Troubleshooting:\n"
            f"  1. Run: python -c \"import ee; ee.Authenticate()\"\n"
            f"  2. Create Cloud project '{project}' at https://console.cloud.google.com/earth-engine\n"
            f"  3. Check network connectivity to *.earthengine.googleapis.com"
        ) from e


if __name__ == "__main__":
    init_ee()
