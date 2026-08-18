from dataclasses import dataclass
from pathlib import Path


@dataclass
class CopernicusDEMLayout:
    dem: Path
    xml: Path | None
    wbm: Path | None
    flm: Path | None
    edm: Path | None
    hem: Path | None

    def metadata_file(self):
        return self.xml


def get_layout(product_dir: str | Path) -> CopernicusDEMLayout:
    product_dir = Path(product_dir)

    return CopernicusDEMLayout(
        dem=next(product_dir.glob("*_DEM.tif")),
        xml=next(product_dir.glob("*.xml"), None),
        wbm=next(product_dir.glob("AUXFILES/*_WBM.tif"), None),
        flm=next(product_dir.glob("AUXFILES/*_FLM.tif"), None),
        edm=next(product_dir.glob("AUXFILES/*_EDM.tif"), None),
        hem=next(product_dir.glob("AUXFILES/*_HEM.tif"), None),
    )
