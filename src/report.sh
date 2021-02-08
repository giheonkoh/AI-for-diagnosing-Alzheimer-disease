echo "Generating report .... "

python -c "import __preprocessing__ as prep; prep.reportinfo('$1','$2','$3')"
