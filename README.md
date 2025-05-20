# Scientific Paper Presentations

This repository contains **markdown-based presentations** for scientific papers, as part of our **Advanced Machine Learning** (INFO8004) course. Each presentation provides a structured summary and key insights from the selected papers.

## Presentations  

1. **[Segment Anything Model (SAM) – Meta](sam-meta.md)**
  - Introduction of necessary **background knowledge**
  - Paper Overview
  - Promptable **Segmentation Task**
  - Model **architecture**
  - Key Particularities
    - solving for **ambiguity**,
    - **attention** layers,
    - **losses** and training algorithm,
    - **zero-shot** learning
  - Discussion (results and limitations)
  - Additional slides on the **SA-1B dataset** and **data engine**   

2. **TBD**  
   - The second presentation will be added soon.  

## How to View the Presentations

- Clone this repo locally: `git clone https://github.com/martinDengis/info8004.git`
- Launch an http server: `python -m http.server 8000`
- Access our presentation on your browser via http://localhost:8000/index.html?p={presentation_name}.md
- _(Enter presentation mode by pressing 'p' key)_

## Contributing

If you’d like to suggest additional paper presentations to this repo, feel free to open a pull request.

## Credits

The markdown-based presentation capability hereby used is largely inspired by classes repos of our professor [@glouppe](https://github.com/glouppe).

## Authors

- [@martinDengis](https://github.com/martinDengis)
- [@giooms](https://github.com/giooms)
