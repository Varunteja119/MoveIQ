from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from datetime import datetime
import argparse
import time

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/ml/raw"
BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"

RAW_DIR.mkdir(parents=True, exist_ok=True)


def filename_for(year, month):
    return f"yellow_tripdata_{year}-{month:02d}.parquet"


def is_valid_parquet(path):
    if not path.exists() or path.stat().st_size < 1_000_000:
        return False

    try:
        with path.open("rb") as file:
            return file.read(4) == b"PAR1"
    except OSError:
        return False


def check_remote_file(year, month):
    filename = filename_for(year, month)
    url = f"{BASE_URL}/{filename}"

    request = Request(
        url,
        headers={"User-Agent": "MoveIQ-Data-Downloader/2.0"},
        method="HEAD",
    )

    try:
        with urlopen(request, timeout=30) as response:
            if response.status != 200:
                return False, None

            size = response.headers.get("Content-Length")
            size = int(size) if size else None

            if size is not None and size < 1_000_000:
                return False, size

            return True, size

    except HTTPError as error:
        if error.code == 404:
            return False, None
        raise

    except URLError:
        raise


def download_file(year, month):
    filename = filename_for(year, month)
    destination = RAW_DIR / filename
    partial = RAW_DIR / f"{filename}.part"
    url = f"{BASE_URL}/{filename}"

    if is_valid_parquet(destination):
        print(f"SKIP: {filename} already exists and is valid")
        return True

    if destination.exists():
        print(f"Invalid existing file: {filename}")
        destination.unlink()

    for attempt in range(1, 4):
        try:
            print(f"\nDownloading {filename} (attempt {attempt}/3)")

            request = Request(
                url,
                headers={"User-Agent": "MoveIQ-Data-Downloader/2.0"},
            )

            downloaded = 0
            last_reported = -1

            with urlopen(request, timeout=120) as response:
                with partial.open("wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break

                        output.write(chunk)
                        downloaded += len(chunk)

                        current_mb = downloaded // (50 * 1024 * 1024)
                        if current_mb > last_reported:
                            print(
                                f"  Downloaded: "
                                f"{downloaded / (1024 * 1024):.1f} MB"
                            )
                            last_reported = current_mb

            if not is_valid_parquet(partial):
                raise ValueError("Downloaded file failed Parquet validation")

            partial.replace(destination)

            print(
                f"SUCCESS: {filename} "
                f"({destination.stat().st_size / (1024 * 1024):.1f} MB)"
            )
            return True

        except Exception as error:
            print(f"FAILED: {error}")

            if partial.exists():
                partial.unlink()

            if attempt < 3:
                time.sleep(3 * attempt)

    print(f"ERROR: Could not download {filename}")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Download available NYC TLC Yellow Taxi Parquet files."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check remote availability without downloading.",
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Check or download only one year.",
    )
    args = parser.parse_args()

    now = datetime.now()
    first_year = args.year if args.year else 2024
    last_year = args.year if args.year else now.year

    if first_year < 2024 or last_year > now.year:
        raise ValueError("Requested year is outside the supported date range.")

    print("=" * 70)
    print("MOVEIQ — TLC YELLOW TAXI DATA DOWNLOADER")
    print("=" * 70)
    print(f"Raw data directory: {RAW_DIR}")
    print(f"Mode: {'CHECK ONLY' if args.check_only else 'DOWNLOAD'}")

    available = []
    unavailable = []
    failed = []
    skipped = 0
    downloaded = 0

    for year in range(first_year, last_year + 1):
        final_month = now.month if year == now.year else 12

        for month in range(1, final_month + 1):
            filename = filename_for(year, month)
            destination = RAW_DIR / filename

            if is_valid_parquet(destination):
                print(f"EXISTS: {filename}")
                skipped += 1
                continue

            if destination.exists():
                print(f"Removing invalid local file: {filename}")
                destination.unlink()

            print(f"Checking TLC availability: {filename}")

            try:
                is_available, remote_size = check_remote_file(year, month)
            except Exception as error:
                print(f"CHECK ERROR: {filename}: {error}")
                failed.append(filename)
                continue

            if not is_available:
                print(f"NOT PUBLISHED: {filename}")
                unavailable.append(filename)
                continue

            size_text = (
                f"{remote_size / (1024 * 1024):.1f} MB"
                if remote_size else "size unknown"
            )
            print(f"AVAILABLE: {filename} ({size_text})")
            available.append(filename)

            if not args.check_only:
                if download_file(year, month):
                    downloaded += 1
                else:
                    failed.append(filename)
                    break

        if failed:
            break

    print("\n" + "=" * 70)
    print("DOWNLOAD / AVAILABILITY SUMMARY")
    print("=" * 70)
    print(f"Already valid locally : {skipped}")
    print(f"Available remotely    : {len(available)}")
    print(f"Downloaded this run   : {downloaded}")
    print(f"Not published         : {len(unavailable)}")
    print(f"Errors                : {len(failed)}")

    total_size = sum(
        path.stat().st_size
        for path in RAW_DIR.glob("yellow_tripdata_*.parquet")
        if is_valid_parquet(path)
    )

    print(f"Total valid local data: {total_size / (1024**3):.2f} GB")

    if unavailable:
        print("\nNot-yet-available months:")
        for name in unavailable:
            print(f"  {name}")

    if failed:
        print("\nFailed months:")
        for name in failed:
            print(f"  {name}")

    print("\nCompleted.")


if __name__ == "__main__":
    main()