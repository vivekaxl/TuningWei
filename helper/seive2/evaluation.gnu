set terminal png
set key horizontal
set output "evaluation.png"
set title "Number of evaluation"
set xlabel "Models"
set ylabel "Evaluations"
set boxwidth 0.15
set logscale y 10
set xrange [-0.5:7]
#set yrange [-6:3]
plot 'evaluation.dat' using ($0-.15):4:5:xtic(1) with boxerrorbars title col, \
'' using ($0):2:3 with boxerrorbars title col,\
'' using ($0+0.15):6:7 with boxerrorbars title col
