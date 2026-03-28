#!/bin/bash
# =============================================================
# ASL CSLR Dataset Download Helper
# =============================================================
# This script assists with downloading required datasets.
# Some downloads require manual browser intervention.
#
# Usage: bash scripts/download_datasets.sh [dataset]
#   Datasets: wlasl, how2sign, ncslgr, all
# =============================================================

set -e

DATA_DIR="$(cd "$(dirname "$0")/../data/raw" && pwd)"

echo "=========================================="
echo "  ASL CSLR Dataset Download Helper"
echo "=========================================="
echo "Data directory: $DATA_DIR"
echo ""

download_wlasl() {
    echo "━━━ WLASL ━━━"
    WLASL_DIR="$DATA_DIR/wlasl"
    
    if [ ! -d "$WLASL_DIR/.git" ]; then
        echo "Cloning WLASL repo..."
        git clone https://github.com/dxli94/WLASL.git "$WLASL_DIR"
    else
        echo "WLASL repo already cloned."
    fi
    
    echo ""
    echo "⚠️  WLASL Video Download:"
    echo "   Most original video URLs have expired."
    echo "   To get the complete WLASL video dataset:"
    echo ""
    echo "   1. Run: cd $WLASL_DIR/start_kit && python3 find_missing.py"
    echo "   2. Submit the Google Form to request pre-processed videos:"
    echo "      https://docs.google.com/forms/d/e/1FAIpQLSc3yHyAranhpkC9ur_Z-Gu5gS5M0WnKtHV07Vo6eL6nZHzruw/viewform"
    echo "   3. You'll receive download links within ~7 days"
    echo "   4. Place the videos in: $WLASL_DIR/start_kit/raw_videos/"
    echo ""
    echo "   The JSON metadata is already available at:"
    echo "   $WLASL_DIR/start_kit/WLASL_v0.3.json"
    echo ""
}

download_how2sign() {
    echo "━━━ How2Sign ━━━"
    H2S_DIR="$DATA_DIR/how2sign"
    mkdir -p "$H2S_DIR/annotations" "$H2S_DIR/keypoints"
    
    # Download annotations (small, works with gdown)
    echo "Downloading annotations..."
    if [ ! -f "$H2S_DIR/annotations/train_annotations.csv" ]; then
        gdown 1dUHSoefk9OxKJnHrHPX--I4tpm9QD0ok -O "$H2S_DIR/annotations/train_annotations.csv"
    fi
    if [ ! -f "$H2S_DIR/annotations/val_annotations.csv" ]; then
        gdown 1Vpag7VPfdTCCJSao8Pz14rlPfekRMggI -O "$H2S_DIR/annotations/val_annotations.csv"
    fi
    if [ ! -f "$H2S_DIR/annotations/test_annotations.csv" ]; then
        gdown 1AgwBZW26kFHS4CWNMQTCMPGkBPkH3qCu -O "$H2S_DIR/annotations/test_annotations.csv"
    fi
    echo "✓ Annotations downloaded"
    
    echo ""
    echo "⚠️  How2Sign 2D Keypoints (large .tar.gz files — may need browser download):"
    echo ""
    echo "   Download these files and place in: $H2S_DIR/keypoints/"
    echo ""
    echo "   Train (21GB) — train_2D_keypoints.tar.gz:"
    echo "   https://drive.google.com/file/d/1TBX7hLraMiiLucknM1mhblNVomO9-Y0r/view"
    echo ""
    echo "   Validation (1.2GB) — val_2D_keypoints.tar.gz:"
    echo "   https://drive.google.com/file/d/1JmEsU0GYUD5iVdefMOZpeWa_iYnmK_7w/view"
    echo ""
    echo "   Test (1.6GB) — test_2D_keypoints.tar.gz:"
    echo "   https://drive.google.com/file/d/1g8tzzW5BNPzHXlamuMQOvdwlHRa-29Vp/view"
    echo ""
    echo "   After downloading, extract:"
    echo "   tar -xf train_2D_keypoints.tar.gz -C $H2S_DIR/keypoints/"
    echo "   tar -xf val_2D_keypoints.tar.gz   -C $H2S_DIR/keypoints/"
    echo "   tar -xf test_2D_keypoints.tar.gz  -C $H2S_DIR/keypoints/"
    echo ""
}

download_ncslgr() {
    echo "━━━ NCSLGR ━━━"
    NCSLGR_DIR="$DATA_DIR/asllrp_ncslgr"
    mkdir -p "$NCSLGR_DIR"
    
    echo "Attempting XML annotations download..."
    if [ ! -f "$NCSLGR_DIR/ncslgr-xml.zip" ]; then
        curl -L --connect-timeout 30 --max-time 300 \
            -o "$NCSLGR_DIR/ncslgr-xml.zip" \
            "http://secrets.rutgers.edu/dai/xml/ncslgr-xml.zip" 2>&1 || {
            echo ""
            echo "⚠️  NCSLGR server may be temporarily unavailable."
            echo "   Try downloading manually:"
            echo "   http://secrets.rutgers.edu/dai/xml/ncslgr-xml.zip"
            echo "   Place in: $NCSLGR_DIR/ncslgr-xml.zip"
        }
    fi
    
    if [ -f "$NCSLGR_DIR/ncslgr-xml.zip" ]; then
        echo "Extracting XML annotations..."
        unzip -o "$NCSLGR_DIR/ncslgr-xml.zip" -d "$NCSLGR_DIR/xml/"
        echo "✓ NCSLGR XML annotations extracted"
    fi
    
    echo ""
    echo "   Video index chart:"
    echo "   https://www.bu.edu/asllrp/ncslgr-for-download/video_index-20120129.zip"
    echo ""
    echo "   Videos are available through the DAI:"
    echo "   http://secrets.rutgers.edu/dai/queryPages/"
    echo ""
    echo "   Python XML parser for SignStream annotations:"
    echo "   https://www.bu.edu/asllrp/ncslgr-for-download/signstream-xmlparser.zip"
    echo ""
}

case "${1:-all}" in
    wlasl)    download_wlasl ;;
    how2sign) download_how2sign ;;
    ncslgr)   download_ncslgr ;;
    all)
        download_wlasl
        download_how2sign
        download_ncslgr
        echo "=========================================="
        echo "  Download Summary"
        echo "=========================================="
        echo ""
        echo "  ✓ WLASL repo cloned (videos need manual request)"
        echo "  ✓ How2Sign annotations downloaded"  
        echo "  ⚠ How2Sign keypoints need browser download (24GB)"
        echo "  ⚠ NCSLGR may need manual download"
        echo ""
        echo "  After downloading, run preprocessing:"
        echo "  python scripts/preprocess.py --dataset <name>"
        echo ""
        ;;
    *)
        echo "Usage: $0 [wlasl|how2sign|ncslgr|all]"
        exit 1
        ;;
esac
