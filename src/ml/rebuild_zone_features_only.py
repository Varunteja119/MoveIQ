import runpy
import duckdb

module = runpy.run_path("src/ml/build_extended_features.py")

con = duckdb.connect()
con.execute("SET memory_limit = '8GB'")
con.execute("SET threads = 4")
con.execute("SET preserve_insertion_order = false")

try:
    module["build_zone_features"](con)
    print("\nZONE FEATURE REBUILD COMPLETED")
finally:
    con.close()
