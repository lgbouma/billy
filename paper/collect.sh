# Usage: copy in figures for the paper

pdir=../results/PTFO_8-8695_results/20200320_linear_amplitude_prior/

cp ${pdir}PTFO_8-8695_transit_2sincosPorb_2sincosProt_sampleplot.png f1.png
cp ${pdir}PTFO_8-8695_transit_2sincosPorb_2sincosProt_splitsignalmap.png f2a.png
cp ${pdir}PTFO_8-8695_transit_2sincosPorb_2sincosProt_splitsignalmap_periodogram.png f2b.png
cp ${pdir}PTFO_8-8695_transit_2sincosPorb_2sincosProt_phasefoldmap.png f3.png
cp ${pdir}PTFO_8-8695_transit_2sincosPorb_2sincosProt_cornerplot.png f4.png
pngquant f4.png --output f4_comp.png
