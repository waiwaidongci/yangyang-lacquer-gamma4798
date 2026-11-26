应用启动时执行 `bootstrap.py`，为 SQLite 建立账号、台账、运行、复核和仪表校准表。重复启动使用同一数据卷；既有数据库在启动时自动为运行表补齐 `calibration_version` 列。
