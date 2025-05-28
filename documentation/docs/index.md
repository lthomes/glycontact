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
- Quantify structural properties such as **SASA** and **flexibility**
- Analyze **conformational preferences** and structure variability
- Compare different glycan structures
- Generate structural features for **machine learning applications**

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

