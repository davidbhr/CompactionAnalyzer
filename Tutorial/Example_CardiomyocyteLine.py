"""Whole-image phalloidin orientation relative to a line along cardiomyocytes.

python Tutorial/Example_CardiomyocyteLine.py --figure cardiomyocyte_line.png
python Tutorial/Example_CardiomyocyteLine.py --saenopy biological_results

Two points define the reference direction, independently of the X and Y axes.
The complete input image is analyzed without a selection mask.
Change REFERENCE_LINE to select a different reference in the image.
"""
from pathlib import Path
import argparse
import sys
import numpy as np
import imageio.v2 as imageio

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from CompactionAnalyzer import analyze_image
from Example_BiologicalOrientation import save_result

DATA=Path(__file__).resolve().parent/'BiologicalExamples/cardiomyocyte_line'
REFERENCE_LINE=((69,469),(655,255))  # (x,y) pixels in the complete input image.


def run(figure=None,saenopy=None,data=DATA):
    data=Path(data)
    source=data/'orientation.tif'
    values=analyze_image(
        source,mode='line_orientation',reference_line=REFERENCE_LINE,
        sigma_tensor=8,edge=15,
        outputs=['angle_line','alignment_line','coherency','selection_mask',
                 'mean_deviation_line','mean_alignment_line','mean_coherency'],
    )
    print({key:value for key,value in values.items() if key.startswith('mean_')})
    if figure is not None:
        save_figure(source,values,figure)
    if saenopy is not None:
        save_result(source,None,values,'cardiomyocyte_line','line_orientation',
                    None,1/6.411017,8.,15,
                    {},saenopy,
                    reference_line=REFERENCE_LINE)
    return values


def save_figure(source,values,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    image=imageio.imread(source)[15:-15,15:-15]
    line=np.asarray(REFERENCE_LINE)-15
    fig,axes=plt.subplots(1,3,figsize=(12.6,4.5),constrained_layout=True)
    for ax in axes[:2]:
        ax.imshow(image,cmap='gray',vmin=0,vmax=np.percentile(image,99.5))
        ax.axline(*line,color='#ffe54d',ls='--',lw=1.3)
        ax.plot(*line.T,'o',color='#ffe54d',mec='black',ms=5)
        ax.set(xlim=(-.5,image.shape[1]-.5),ylim=(image.shape[0]-.5,-.5))
        ax.axis('off')
        ax.plot([30,30+20*6.411017],[image.shape[0]-55]*2,color='white',lw=3)
        ax.text(30,image.shape[0]-75,'20 µm',color='white',fontsize=9)
    mapped=axes[1].imshow(values['alignment_line'],cmap='coolwarm',vmin=-1,vmax=1,alpha=.8)
    axes[0].set_title('Cardiomyocytes · phalloidin')
    axes[1].set_title('Alignment to selected line')
    bar=fig.colorbar(mapped,ax=axes[1],fraction=.045,pad=.02,ticks=[-1,0,1])
    bar.set_label('Alignment')
    angle=values['angle_line'];finite=angle[np.isfinite(angle)]
    counts,bins=np.histogram(finite,bins=np.arange(-90,91,2))
    probability=counts/counts.sum()
    axes[2].bar((bins[:-1]+bins[1:])/2,probability,width=2,color='#8b5db7',edgecolor='white',lw=.3)
    axes[2].axvline(0,color='#444444',lw=1,ls='--')
    axes[2].set(xlim=(-90,90),xlabel='Angle to reference line (°)',ylabel='Probability per bin',
                title=f'2° bins · sum = {probability.sum():.2f}')
    axes[2].spines[['top','right']].set_visible(False)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=170,facecolor='white');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure',type=Path)
    parser.add_argument('--saenopy',type=Path)
    parser.add_argument('--data',type=Path,default=DATA)
    args=parser.parse_args()
    run(args.figure,args.saenopy,args.data)
