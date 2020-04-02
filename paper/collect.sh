# Usage: copy in figures for the paper

pdir=../results/PTFO_8-8695_results/20200320_linear_amplitude_prior/

#cp ${pdir}PTFO_8-8695_transit_2sincosPorb_2sincosProt_splitsignalmap_joined.pdf f1.pdf
cp ${pdir}PTFO_8-8695_transit_2sincosPorb_2sincosProt_splitsignalmap.png f1.png

cp ${pdir}PTFO_8-8695_transit_2sincosPorb_2sincosProt_phasefoldmap.pdf f2.pdf

cp ../results/PTFO_8-8695_results/scene.pdf f3.pdf

cp ../results/ephemeris/O_minus_C.pdf f4.pdf

cp ${pdir}PTFO_8-8695_transit_2sincosPorb_2sincosProt_cornerplot.png f5.png
pngquant f5.png --output f5_comp.png
