# About the project

In this project I am going to implement and evaluate the different strategies for data augmeantion and confident learning in our NLU training data.

The code could achieve the following objectives
+ Data Augmentation in Transformers Library with T5 and GPT3 models.
+ Remove misclassified data with confident learning
+ Training and Testing NLU classifers based on the modeified datasets(Data augmenation with different sizes/Data augmenation and Conifident learning)
+ Evaluate the different Data augmenation and confident learning stratgies based on the performance of relative NLU classifier   

Credits go to :

+ [Karnaj](https://github.com/Karnaj), for the original template and many improvements.
+ [Heziode](https://github.com/Heziode), for later improvements.
+ [pierre-24](https://github.com/pierre-24), maintainer of the repository.

# Before you strat

## Excute in Google Colab

To get a better demonstration to my work, I remcommend you open those scripts in Google Colab

Colab can load public github notebooks directly, with no required authorization step.

For example, consider the notebook at this address: https://github.com/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb.

The direct colab link to this notebook is: https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb.

To generate such links in one click, you can use the [Open in Colab](https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo) Chrome extension.

## installation
Before you start, here are some of required packages you might need to install later. But dont worry, I provide the installation code in my Google Colab notebook as well. you just need to simply click run.

+ Windows: [MikTeX](https://miktex.org/download)
+ Mac OS X: [MacTeX](https://www.tug.org/mactex/mactex-download.html)
+ Linux: [TeXLive](https://tug.org/texlive/) is probably available in your favorite package manager.

Since this template uses the [minted](https://github.com/gpoore/minted/) package you also need [Pygments](http://pygments.org/), probably available in your package manager on Linux or via `pip`:

```bash
pip install Pygments
```

You also need these fonts:

- [Source Code Pro](https://www.fontsquirrel.com/fonts/source-code-pro)
- [Source Sans Pro](https://www.fontsquirrel.com/fonts/source-sans-pro)

## Other images formats

To be able to use GIF and SVG images in your documents, two extra programs are needed:

+ [librsvg](https://github.com/GNOME/librsvg), which uses cairo to convert svg to pdf, and is available under the name `librsvg2-bin` in many package managers ; 
+ The `convert` program, part of the [imagemagick](http://www.imagemagick.org/) tools suite, to convert GIF to PNG. It is probably also available in your package manager.

## Package installation

If you are a developer wanting to help, you can clone this package anywhere and just use the Makefile to run the tests (see [`CONTRIBUTING.md`](./CONTRIBUTING.md)).

To use this package normally, you need to clone it into your `TXMFHOME/tex/latex/` directory (you can know to which location `TXMFHOME` corresponds by running `kpsewhich -var-value TEXMFHOME`, but probably `$HOME/texmf/`). 
Note that you don't need to run `texhash`. 
More information is given for example [here](https://faculty.math.illinois.edu/~hildebr/tex/tips-customstyles.html).

This repo uses submodules. After clone this repo, in root folder of the project, execute this command to download the submodules: `git submodule update --init --recursive`.

Note that this package requires `lualatex` to be called with the `-shell-escape` option (because of minted).

# Testing and using

The different macros and environment are defined in [`zmdocument.cls`](./zmdocument.cls) and documented in [`documentation.md`](./documentation.md).
Here is a skeleton on what your LaTeX document should contain:

```latex
\documentclass{zmdocument}

\title{Title}
\author{Author}
\licence[path/to/image]{Licence name}{URL} % optional
\logo{logo.png}  % if ./logo.png is available

\begin{document}
\maketitle
\tableofcontents

%% ... The rest of your document
\end{document}
```

See [the `test.tex` file in tests](./tests/test.tex) for an example usage of the document class.


# Contributing

See [`CONTRIBUTING.md`](./CONTRIBUTING.md).















<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Here's a blank template to get started:
**To avoid retyping too much info. Do a search and replace with your text editor for the following:**
`github_username`, `repo_name`, `twitter_handle`, `email`, `project_title`, `project_description`


### Built With

* []()
* []()
* []()



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
2. Install NPM packages
   ```sh
   npm install
   ```



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/github_username/repo_name/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* []()
* []()
* []()
