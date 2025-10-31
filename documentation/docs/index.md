# GlyContact: 3D Analysis of Glycan Structures
---
**GlyContact** is a Python package for retrieving, processing, and analyzing 3D glycan structures from GlycoShape, molecular dynamics, NMR, or X-ray crystallography.

The package is organized into the following main modules:

- `process`: utilities for parsing and analyzing 3D glycan structures
- `visualize`: functions for plotting contact maps and glycan features
- `learning`: functions for training and using machine learning models

<br>

GlyContact provides a comprehensive toolkit that enables researchers to:

- Visualize complex glycan structures with **3D-SNFG symbols**
- Quantify structural properties including **solvent accessible surface area** (SASA), **motif flexibility** and **torsion angles**
- Analyze **relationships** between glycan composition/sequence/class and these structural properties
- **Compare** different glycan structures that are either free, bound to lectins or covalently linked to glycoproteins 
- Generate structural features for** machine learning applications**

These capabilities help bridge the gap between **glycan sequence** and **function** by revealing the critical spatial arrangements that determine molecular recognition.

<br><br>


## **Install**
---
**GlyContact** can be cloned from GitHub or directly installed using **pip**.

All modules in **GlyContact**, except for ml, can be run on any machine. For most parts of ml, however, a GPU is needed to load and run **torch_geometric**.

<br>

### **Requirements**
---
We recommend using at least the following Python and packages versions to ensure similar functionalities and performances as described in the publication: 

- **Python** ≥ 3.12.6 
- **glycowork** ≥ 1.6 
- **scipy** ≥ 1.11

<br>

### **Installation using pip**
---
If you are using pip, all the required Python packages will be automatically installed with GlyContact.

```bash
pip install git+https://github.com/lthomes/glycontact.git
```

<br>

An optional `[ml]` install is available for machine learning features:

```bash
pip install -e git+https://github.com/lthomes/glycontact.git#egg=glycontact[ml]
```

<br>

### **Getting started with GlyContact**

**GlyContact** enables detailed exploration of glycan conformational properties by connecting glycan 3D structural data with their corresponding sequence information.

By default, users can input glycan sequences, which are automatically mapped to the appropriate conformer structures from **GlycoShape** through built-in functions.

To streamline structural analysis, **GlyContact** integrates directly with the GlycoShape API, allowing automatic retrieval of structures and eliminating the challenges associated with manual data downloads.

Alternatively, **GlyContact** can operate independently of GlycoShape if users prefer to supply their own structural data by providing file paths to local PDB files.

<br><br>

## **Contributing**
---
Contributions to GlyContact are welcome! Please feel free to submit a Pull Request.

<br><br>

## **Citation**
---
If you use GlyContact in your research, please cite:

```[Citation information will be added upon publication]```

<br><br>

## **Licence**
---
This project is licensed under the MIT License—see the LICENSE file for details.

```  ```

