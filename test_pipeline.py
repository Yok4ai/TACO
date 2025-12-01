#!/usr/bin/env python3
"""
Quick test script to validate the TACO RF-DETR pipeline setup
"""

import sys
import json
from pathlib import Path


def test_file_structure():
    """Test if all required files are present"""
    print("Testing file structure...")

    base_dir = Path(__file__).parent
    required_files = [
        'prepare_taco_dataset.py',
        'taco_rfdetr_lightning.py',
        'train_taco_rfdetr.py',
        'inference_taco_rfdetr.py',
        'requirements.txt',
        'README.md',
        '.env'
    ]

    missing_files = []
    for file in required_files:
        if not (base_dir / file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True


def test_data_structure():
    """Test if TACO data structure is correct"""
    print("\nTesting TACO data structure...")

    data_dir = Path('/home/mkultra/Documents/TACO/TACO/data')

    if not data_dir.exists():
        print("❌ Data directory not found")
        return False

    # Check for annotations file
    annotations_file = data_dir / 'annotations.json'
    if not annotations_file.exists():
        print("❌ annotations.json not found")
        return False

    # Check batch directories
    batch_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('batch_')]
    if len(batch_dirs) == 0:
        print("❌ No batch directories found")
        return False

    print(f"✅ Found {len(batch_dirs)} batch directories")

    # Test loading annotations
    try:
        with open(annotations_file, 'r') as f:
            data = json.load(f)

        print(f"✅ Annotations loaded successfully")
        print(f"   - Images: {len(data.get('images', []))}")
        print(f"   - Annotations: {len(data.get('annotations', []))}")
        print(f"   - Categories: {len(data.get('categories', []))}")

        return True

    except Exception as e:
        print(f"❌ Error loading annotations: {e}")
        return False


def test_environment():
    """Test environment configuration"""
    print("\nTesting environment...")

    env_file = Path('/home/mkultra/Documents/TACO/TACO/.env')
    if not env_file.exists():
        print("❌ .env file not found")
        return False

    try:
        with open(env_file, 'r') as f:
            content = f.read()

        if 'ROBOFLOW_API_KEY' in content:
            print("✅ ROBOFLOW_API_KEY found in .env")
            return True
        else:
            print("❌ ROBOFLOW_API_KEY not found in .env")
            return False

    except Exception as e:
        print(f"❌ Error reading .env file: {e}")
        return False


def test_python_syntax():
    """Test if Python scripts have valid syntax"""
    print("\nTesting Python syntax...")

    base_dir = Path(__file__).parent
    python_files = [
        'prepare_taco_dataset.py',
        'taco_rfdetr_lightning.py',
        'train_taco_rfdetr.py',
        'inference_taco_rfdetr.py'
    ]

    for file in python_files:
        try:
            with open(base_dir / file, 'r') as f:
                code = f.read()

            compile(code, file, 'exec')
            print(f"✅ {file} - syntax OK")

        except SyntaxError as e:
            print(f"❌ {file} - syntax error: {e}")
            return False
        except Exception as e:
            print(f"❌ {file} - error: {e}")
            return False

    return True


def generate_usage_instructions():
    """Generate usage instructions"""
    print("\n" + "="*60)
    print("TACO RF-DETR Pipeline Setup Complete!")
    print("="*60)

    print("\nNext steps:")
    print("1. Install dependencies:")
    print("   pip install -r requirements.txt")

    print("\n2. Prepare the dataset:")
    print("   python3 prepare_taco_dataset.py")

    print("\n3. Train the model:")
    print("   python3 train_taco_rfdetr.py --model_size medium --batch_size 8 --max_epochs 50")

    print("\n4. Run inference:")
    print("   python3 inference_taco_rfdetr.py --input path/to/image.jpg --model_path checkpoints/best_model.ckpt")

    print("\nFor more details, see README.md")


def main():
    """Main test function"""
    print("TACO RF-DETR Pipeline Validation")
    print("=" * 40)

    tests = [
        test_file_structure,
        test_data_structure,
        test_environment,
        test_python_syntax
    ]

    all_passed = True
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
            all_passed = False

    print("\n" + "="*40)
    if all_passed:
        print("✅ All tests passed!")
        generate_usage_instructions()
    else:
        print("❌ Some tests failed. Please check the issues above.")
        sys.exit(1)


if __name__ == '__main__':
    main()