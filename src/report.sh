echo "Generating report .... "

python3 -c "import __preprocessing__ as prep; prep.reportinfo('$1','$2','$3')"

echo ""
echo "... Done! "
