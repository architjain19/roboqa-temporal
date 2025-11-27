from pathlib import Path
from roboqa_temporal.calibration import CalibrationQualityValidator, CalibrationStream

def make_synthetic_pair(name, miscalib_px):
    # Create a dummy stream with encoded miscalibration in the filename
    return CalibrationStream(
        name=name,
        image_paths=["/dummy/img.png"],
        pointcloud_paths=["/dummy/cloud.bin"],
        calibration_file=f"/dummy/calib_{name}_miscalib_{miscalib_px}px.txt",
        camera_id="cam0",
        lidar_id="lidar0"
    )

def main():
    # Output directory relative to where we run the script
    output_dir = Path("examples/sample_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    validator = CalibrationQualityValidator(output_dir=str(output_dir))
    
    # Create one good pair (0.5px error) and one bad pair (15.0px error)
    pairs = {
        "front_camera_good": make_synthetic_pair("front", 0.5),
        "side_camera_bad": make_synthetic_pair("side", 15.0),
    }
    
    print(f"Generating report in {output_dir.absolute()}...")
    report = validator.analyze_sequences(pairs, bag_name="sample_run")
    print(f"Generated HTML report: {report.html_report_file}")

if __name__ == "__main__":
    main()
