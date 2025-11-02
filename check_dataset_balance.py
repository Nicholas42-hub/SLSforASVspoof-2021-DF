"""
Quick script to check label distribution in protocol file
Run this before visualization to ensure balanced sampling is possible
"""

import argparse
from collections import Counter
import os
import glob

def find_protocol_files(base_dir):
    """Find potential protocol files in a directory"""
    print(f"🔍 Searching for protocol files in: {base_dir}")
    
    # Common protocol file patterns
    patterns = [
        '*.txt',
        '*protocol*.txt',
        '*cm*.txt',
        '*eval*.txt',
        '*trial*.txt'
    ]
    
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(base_dir, pattern)))
        files.extend(glob.glob(os.path.join(base_dir, '**', pattern), recursive=True))
    
    # Remove duplicates
    files = list(set(files))
    
    if files:
        print(f"📁 Found {len(files)} potential protocol files:")
        for f in sorted(files):
            size_kb = os.path.getsize(f) / 1024
            print(f"   - {f} ({size_kb:.1f} KB)")
    else:
        print(f"   ❌ No .txt files found")
    
    return files

def check_protocol_balance(protocol_path):
    """Parse protocol file and show label distribution"""
    print(f"📋 Checking protocol file: {protocol_path}", flush=True)
    
    import os
    if not os.path.exists(protocol_path):
        print(f"❌ Error: File not found: {protocol_path}", flush=True)
        return 0, 0
    
    # Show file info
    file_size = os.path.getsize(protocol_path)
    print(f"📁 File size: {file_size / 1024:.1f} KB", flush=True)
    
    labels = []
    line_count = 0
    parse_errors = 0
    
    # Try to detect format by reading first few lines
    print(f"🔍 Detecting file format...", flush=True)
    with open(protocol_path, 'r') as f:
        first_lines = [f.readline() for _ in range(5)]
    
    print(f"📝 First line sample: {first_lines[0].strip()[:100]}", flush=True)
    print(f"📝 Line parts count: {len(first_lines[0].split())}", flush=True)
    
    # Check if it's just a file list (no labels)
    if len(first_lines[0].split()) == 1:
        print(f"⚠️  WARNING: This appears to be a file list, not a protocol file!")
        print(f"   Protocol files should have labels (bonafide/spoof)")
        print(f"   This file only contains file IDs.\n")
        
        # Try to find actual protocol files
        base_dir = os.path.dirname(protocol_path)
        if base_dir:
            potential_files = find_protocol_files(base_dir)
            if potential_files:
                print(f"\n💡 Try one of these files instead:")
                for pf in sorted(potential_files):
                    if 'cm' in pf.lower() and 'eval' in pf.lower():
                        print(f"   ⭐ {pf}  (recommended - contains 'cm' and 'eval')")
                    else:
                        print(f"      {pf}")
        
        return 0, 0
    
    with open(protocol_path, 'r') as f:
        for line in f:
            line_count += 1
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            
            # Try different format possibilities
            parsed = False
            
            # Format 0A: trial_metadata.txt format - bonafide lines
            # LA_0007-alaw-ita_tx LA_E_5013670-alaw-ita_tx alaw ita_tx bonafide nontarget notrim eval
            # Column 4: 'bonafide'
            if len(parts) >= 5 and parts[4] == 'bonafide':
                labels.append(1)
                parsed = True
            
            # Format 0B: trial_metadata.txt format - spoof lines  
            # LA_0013-alaw-ita_tx LA2021-LA_E_5658320 alaw ita_tx A07 spoof notrim progress
            # Column 4: attack type (A07, A08, etc.), Column 5: 'spoof'
            elif len(parts) >= 6 and parts[5] == 'spoof':
                labels.append(0)
                parsed = True
            
            # Format 0C: Original trial_metadata.txt (column 4 can be bonafide or spoof)
            elif len(parts) >= 6:
                label_str = parts[4]  # Label at column 4
                if label_str == 'bonafide':
                    labels.append(1)
                    parsed = True
                elif label_str == 'spoof':
                    labels.append(0)
                    parsed = True
            
            # Format 1: speaker_id file_id - label attack_type (5+ parts)
            if not parsed and len(parts) >= 5 and parts[2] == '-':
                label_str = parts[3]
                if label_str == 'bonafide':
                    labels.append(1)
                    parsed = True
                elif label_str == 'spoof':
                    labels.append(0)
                    parsed = True
            
            # Format 2: speaker_id file_id label attack_type (4+ parts, no dash)
            elif not parsed and len(parts) >= 4:
                label_str = parts[2]
                if label_str == 'bonafide':
                    labels.append(1)
                    parsed = True
                elif label_str == 'spoof':
                    labels.append(0)
                    parsed = True
            
            # Format 3: file_id label (2 parts)
            elif not parsed and len(parts) >= 2:
                label_str = parts[1]
                if label_str == 'bonafide':
                    labels.append(1)
                    parsed = True
                elif label_str == 'spoof':
                    labels.append(0)
                    parsed = True
            
            if not parsed:
                parse_errors += 1
                if parse_errors <= 3:
                    print(f"⚠️  Could not parse line {line_count}: {line[:80]}")
    
    print(f"✅ Processed {line_count} lines, parsed {len(labels)} samples, {parse_errors} errors")
    
    # Count distribution
    counter = Counter(labels)
    bonafide_count = counter.get(1, 0)
    spoof_count = counter.get(0, 0)
    total = len(labels)
    
    print(f"\n📊 Label Distribution:")
    print(f"   Total samples: {total}")
    
    if total == 0:
        print(f"   ❌ No valid samples found!")
        print(f"   💡 Check if the file format matches expected patterns:")
        print(f"      - Format 1: speaker_id file_id - bonafide/spoof attack_type")
        print(f"      - Format 2: speaker_id file_id bonafide/spoof attack_type")
        print(f"      - Format 3: file_id bonafide/spoof")
        return 0, 0
    
    print(f"   Bonafide (label=1): {bonafide_count} ({bonafide_count/total*100:.2f}%)")
    print(f"   Spoof (label=0): {spoof_count} ({spoof_count/total*100:.2f}%)")
    
    # Check if balanced sampling is feasible
    print(f"\n🎯 Balanced Sampling Feasibility:")
    if bonafide_count == 0:
        print(f"   ❌ No bonafide samples - balanced sampling will fail!")
    elif spoof_count == 0:
        print(f"   ❌ No spoof samples - balanced sampling will fail!")
    else:
        max_balanced = min(bonafide_count, spoof_count) * 2
        print(f"   ✅ Can collect up to {max_balanced} balanced samples")
        print(f"      ({min(bonafide_count, spoof_count)} from each class)")
    
    return bonafide_count, spoof_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check label distribution in ASVspoof protocol files')
    parser.add_argument('--protocols_path', type=str, required=False, 
                       help='Path to protocol file')
    parser.add_argument('--search_dir', type=str, required=False,
                       help='Directory to search for protocol files')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔍 ASVspoof Protocol File Checker")
    print("=" * 70)
    
    if args.search_dir:
        # Just search for files
        find_protocol_files(args.search_dir)
    elif args.protocols_path:
        # Check specific file
        check_protocol_balance(args.protocols_path)
    else:
        print("❌ Please provide either --protocols_path or --search_dir")
        print("\nUsage examples:")
        print("  python check_dataset_balance.py --protocols_path /path/to/trial_metadata.txt")
        print("  python check_dataset_balance.py --search_dir /path/to/dataset/")
        parser.print_help()
