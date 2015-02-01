set terminal png
set key horizontal
set xlabel "Models"
set ylabel "Score"
set title "Scores"
set output "score.png"
set boxwidth 0.15
set xrange [-0.5:7]
set yrange [-6:3]
plot 'score.dat' using ($0-.15):4:5:xtic(1) with boxerrorbars title col, \
'' using ($0):2:3 with boxerrorbars title col,\
'' using ($0+0.15):6:7 with boxerrorbars title col
