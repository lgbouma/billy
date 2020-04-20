# Usage: copy in figures for the paper

pdir=../results/PTFO_8-8695_results/20200413_v0/

cp ${pdir}PTFO_8-8695_transit_2sincosPorb_2sincosProt_splitsignalmap_i.pdf f1.pdf
cp ${pdir}PTFO_8-8695_transit_2sincosPorb_2sincosProt_splitsignalmap_ii.pdf f2.pdf

cp ${pdir}PTFO_8-8695_transit_2sincosPorb_2sincosProt_phasefoldmap.pdf f3.pdf

cp ../results/PTFO_8-8695_results/scene.pdf f4.pdf

cp ../results/cluster_membership/hr.pdf f5a.pdf
cp ../results/cluster_membership/astrometric_excess.pdf f5b.pdf

cp ../results/ephemeris/O_minus_C.pdf f6.pdf

cp ../results/brethren/brethren.pdf f7.pdf

# cp ${pdir}PTFO_8-8695_transit_2sincosPorb_2sincosProt_cornerplot.png f6.png
# pngquant f6.png --output f6_comp.png
