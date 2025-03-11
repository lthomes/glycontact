# GlyContact: 3D Analysis of Glycan Structures

GlyContact is a Python package for retrieving, processing, and analyzing 3D glycan structures from GlycoShape/molecular dynamics, NMR, or X-ray crystallography. While glycans are traditionally represented as linear text sequences, their branched structures and high flexibility create a complex 3D landscape that affects their biological function.

GlyContact provides a comprehensive toolkit for analyzing 3D glycan structures, enabling researchers to:

1. Visualize complex glycan structures with 3D-SNFG symbols
2. Quantify structural properties such as SASA and flexibility
3. Analyze conformational preferences and structure variability
4. Compare different glycan structures
5. Generate structural features for machine learning applications

These capabilities help bridge the gap between glycan sequence and function by revealing the critical spatial arrangements that determine molecular recognition.

## Installation

```bash
pip install git+https://github.com/lthomes/glycontact/
```

## 3D Visualization of Glycan Structures

GlyContact can visualize 3D glycan structures with Symbol Nomenclature for Glycans (SNFG) representation, making it easy to identify different monosaccharides in complex structures.

Let's visualize a glycan:


```python
from glycontact.visualize import plot_glycan_3D
plot_glycan_3D("Fuc(a1-2)Gal(b1-3)GalNAc", show_volume=True)
```


<div id="3dmolviewer_17416707793408606"  style="position: relative; width: 800px; height: 800px;">
        <p id="3dmolwarning_17416707793408606" style="background-color:#ffcccc;color:black">3Dmol.js failed to load for some reason.  Please check your browser console for error messages.<br></p>
        </div>
<script>

var loadScriptAsync = function(uri){
  return new Promise((resolve, reject) => {
    //this is to ignore the existence of requirejs amd
    var savedexports, savedmodule;
    if (typeof exports !== 'undefined') savedexports = exports;
    else exports = {}
    if (typeof module !== 'undefined') savedmodule = module;
    else module = {}

    var tag = document.createElement('script');
    tag.src = uri;
    tag.async = true;
    tag.onload = () => {
        exports = savedexports;
        module = savedmodule;
        resolve();
    };
  var firstScriptTag = document.getElementsByTagName('script')[0];
  firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
});
};

if(typeof $3Dmolpromise === 'undefined') {
$3Dmolpromise = null;
  $3Dmolpromise = loadScriptAsync('https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.0/3Dmol-min.js');
}

var viewer_17416707793408606 = null;
var warn = document.getElementById("3dmolwarning_17416707793408606");
if(warn) {
    warn.parentNode.removeChild(warn);
}
$3Dmolpromise.then(function() {
viewer_17416707793408606 = $3Dmol.createViewer(document.getElementById("3dmolviewer_17416707793408606"),{backgroundColor:"white"});
viewer_17416707793408606.zoomTo();
	viewer_17416707793408606.addModel("REMARK #################################################################\nREMARK #################################################################\nREMARK #################################################################\nREMARK    Restoring Protein Glycosylation with GlycoShape\nREMARK    Callum M Ives,*, Ojas Singh,*, Silvia D\u00e2\u20ac\u2122Andrea, Carl A Fogarty, Aoife M Harbison, Akash Satheesan, Beatrice Tropea, Elisa Fadda\nREMARK    bioRxiv, 2023\nREMARK ################################################################\nREMARK ################################################################\nREMARK ################################################################\nATOM      2  O1  ROH X   1      23.141  19.692  16.830  1.00  0.00      SYST O  \nATOM      3  C1  A2G X   2      21.875  20.368  16.984  1.00  0.00      SYST C  \nATOM      5  C2  A2G X   2      21.541  20.701  18.489  1.00  0.00      SYST C  \nATOM      7  N2  A2G X   2      22.541  21.597  19.217  1.00  0.00      SYST N  \nATOM      9  C2N A2G X   2      22.403  22.923  19.216  1.00  0.00      SYST C  \nATOM     10  CME A2G X   2      23.497  23.801  19.766  1.00  0.00      SYST C  \nATOM     14  O2N A2G X   2      21.369  23.465  18.744  1.00  0.00      SYST O  \nATOM     15  C3  A2G X   2      21.248  19.431  19.307  1.00  0.00      SYST C  \nATOM     17  C4  A2G X   2      20.160  18.500  18.708  1.00  0.00      SYST C  \nATOM     19  C5  A2G X   2      20.463  18.295  17.158  1.00  0.00      SYST C  \nATOM     21  C6  A2G X   2      19.458  17.386  16.449  1.00  0.00      SYST C  \nATOM     24  O6  A2G X   2      20.056  16.779  15.264  1.00  0.00      SYST O  \nATOM     26  O5  A2G X   2      20.736  19.581  16.450  1.00  0.00      SYST O  \nATOM     27  O4  A2G X   2      18.875  19.088  18.844  1.00  0.00      SYST O  \nATOM     29  O3  A2G X   2      20.631  19.853  20.595  1.00  0.00      SYST O  \nATOM     30  C1  GAL X   3      21.614  19.789  21.726  1.00  0.00      SYST C  \nATOM     32  O5  GAL X   3      21.676  18.401  22.302  1.00  0.00      SYST O  \nATOM     33  C5  GAL X   3      22.766  18.169  23.284  1.00  0.00      SYST C  \nATOM     35  C6  GAL X   3      22.721  16.670  23.786  1.00  0.00      SYST C  \nATOM     38  O6  GAL X   3      21.353  16.217  24.052  1.00  0.00      SYST O  \nATOM     40  C4  GAL X   3      22.604  19.250  24.457  1.00  0.00      SYST C  \nATOM     42  O4  GAL X   3      21.490  19.001  25.331  1.00  0.00      SYST O  \nATOM     44  C3  GAL X   3      22.570  20.698  23.850  1.00  0.00      SYST C  \nATOM     46  O3  GAL X   3      22.558  21.652  24.899  1.00  0.00      SYST O  \nATOM     48  C2  GAL X   3      21.330  20.790  22.902  1.00  0.00      SYST C  \nATOM     50  O2  GAL X   3      21.184  22.202  22.401  1.00  0.00      SYST O  \nATOM     51  C1  FUC X   4      19.926  22.769  22.912  1.00  0.00      SYST C  \nATOM     53  O5  FUC X   4      18.777  22.056  22.299  1.00  0.00      SYST O  \nATOM     54  C5  FUC X   4      18.108  22.379  21.064  1.00  0.00      SYST C  \nATOM     56  C6  FUC X   4      16.688  21.777  21.084  1.00  0.00      SYST C  \nATOM     60  C4  FUC X   4      18.115  23.924  20.870  1.00  0.00      SYST C  \nATOM     62  O4  FUC X   4      17.120  24.523  21.619  1.00  0.00      SYST O  \nATOM     64  C3  FUC X   4      19.442  24.698  21.267  1.00  0.00      SYST C  \nATOM     66  O3  FUC X   4      19.334  26.141  21.178  1.00  0.00      SYST O  \nATOM     68  C2  FUC X   4      19.850  24.336  22.700  1.00  0.00      SYST C  \nATOM     70  O2  FUC X   4      21.132  24.998  23.039  1.00  0.00      SYST O  \nTER      72      FUC X   4                                                       \nEND   \n","pdb");
	viewer_17416707793408606.addSurface(1,{"opacity": 0.5});
	viewer_17416707793408606.addBox({"center": {"x": 21.003833333333333, "y": 19.479333333333333, "z": 17.849333333333334}, "dimensions": {"w": 1.0, "h": 1.0, "d": 1.0}, "color": "#FCC326", "alpha": 1.0});
	viewer_17416707793408606.addSphere({"center": {"x": 22.09333333333333, "y": 19.516166666666667, "z": 23.086833333333335}, "radius": 0.7, "color": "#FCC326", "alpha": 1.0});
	viewer_17416707793408606.addArrow({"start": {"x": 19.036333333333335, "y": 23.360333333333333, "z": 21.852}, "end": {"x": 19.036333333333335, "y": 24.560333333333332, "z": 21.852}, "radius": 0.4, "mid": 0.01, "color": "#C23537", "alpha": 1.0, "resolution": 32});
	viewer_17416707793408606.setStyle({"stick": {}});
	viewer_17416707793408606.zoomTo();
	viewer_17416707793408606.render();
viewer_17416707793408606.render();
});
</script>





    <py3Dmol.view at 0x1ed48815be0>



## Glycan Contact Maps

Contact maps reveal the spatial relationships between monosaccharides in a glycan structure. These maps help identify which parts of the glycan are in close proximity, providing insights into potential functional regions.


```python
from glycontact.process import get_contact_tables
# Get monosaccharide contact tables
glycan = "Gal(b1-4)GlcNAc(b1-2)Man(a1-3)[Gal(b1-4)GlcNAc(b1-2)Man(a1-6)]Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc"
contact_tables = get_contact_tables(glycan, level="monosaccharide")

from glycontact.visualize import draw_contact_map
# Visualize the first contact map
draw_contact_map(contact_tables[0], size=1.0)
```


    
![png](README_files/README_6_0.png)
    


## Surface Accessibility and Flexibility

The solvent-accessible surface area (SASA) and flexibility of monosaccharides are crucial determinants of glycan-protein interactions. GlyContact calculates these properties and allows visualization of their distribution across the glycan structure.


```python
from glycontact.visualize import plot_glycan_score
plot_glycan_score(glycan, attribute="SASA")
```




    
![svg](README_files/README_8_0.svg)
    



## Glycosidic Torsion Angles

Glycosidic torsion angles (phi/psi) determine the overall shape of glycans. GlyContact can analyze these angles across multiple structures to identify preferred conformations, similar to protein Ramachandran plots.


```python
from glycontact.visualize import ramachandran_plot
ramachandran_plot("GlcNAc(b1-4)GlcNAc")
```


    
![png](README_files/README_10_0.png)
    


## Contributing

Contributions to GlyContact are welcome! Please feel free to submit a Pull Request.

## Citation

If you use GlyContact in your research, please cite:

```
[Citation information will be added upon publication]
```

## License

This project is licensed under the MIT License—see the LICENSE file for details.


```python

```
