%% %% Public latex format for SMAI-JCM
%%% DO NOT MODIFY THIS FILE
%%%%
\ProvidesClass{cedram-smai-jcm}[2008/01/01 v0.2]
%% Option langue
\newif\if@francais\@francaisfalse
\DeclareOption{francais}{\@francaistrue}
\DeclareOption{anglais}{\@francaisfalse}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{amsart}}
\ProcessOptions\relax
%% Using amsart
\LoadClass[centertags,portrait,11pt,twoside,%
reqno,onecolumn]{amsart}[2004/08/06 v2.20]
%% changing amsart for cedram compatibility
\AtBeginDocument{%
  \@for\@tempa:=-1,0,1,2,3,4\do{%
    \@ifundefined{r@tocindent\@tempa}{%
      \@xp\gdef\csname r@tocindent\@tempa\endcsname{0pt}}{}%
  }%
}
\def\@writetocindents{%
  \begingroup
  \@for\@tempa:=-1,0,1,2,3\do{%
    \immediate\write\@auxout{%
      \string\newlabel{tocindent\@tempa}{%
        \csname r@tocindent\@tempa\endcsname}}%
  }%
  \endgroup}
%% The journal is typeset on A4 pages, this is a hack to get the
%% pages printable on letter if your paper defaults to letter in your
%% local TeX installation  
\AtBeginDocument{%
\ifdim\pdfpageheight>11in % probably A4
\else %probably Letter
\voffset-15mm
\fi}


\newif\ifcdr@hyperref
%% change if incompatible with hyperref
\cdr@hyperreftrue
\ifcdr@hyperref
\RequirePackage{hyperref}\fi
\RequirePackage{graphicx}
%  \author macro revisited
\renewcommand{\author}[2][]{%
  \ifx\@empty\authors
    \gdef\authors{#2}%
    \g@addto@macro\addresses{\author{#2}}%
  \else
    \g@addto@macro\authors{\and#2}%
    \g@addto@macro\addresses{\author{#2}}%
  \fi
  \@ifnotempty{#1}{%
    \ifx\@empty\shortauthors
      \gdef\shortauthors{#1}%
    \else
      \g@addto@macro\shortauthors{\and#1}%
    \fi
  }%
}
\edef\author{\@nx\@dblarg
  \@xp\@nx\csname\string\author\endcsname}
\renewcommand{\address}[2][]{\g@addto@macro\addresses{\address{#1}{#2}}\g@addto@macro\authors{\hasaddress{#1}{#2}}}
%% Predefined theorems
\newif\ifcdr@thmsin\cdr@thmsintrue
\newif\ifcdr@thmsas\cdr@thmsasfalse
\newif\ifcdr@thm@necounter\cdr@thm@necounterfalse
\def\OneNumberAllTheorems{\cdr@thm@necountertrue}
\def\NumberTheoremsIn#1{\@ifempty{#1}%
     {\cdr@thmsinfalse}%
     {\@ifundefined{c@#1}{\@nocounterr{#1}}%
       {\cdr@thmsintrue\cdr@thmsasfalse\def\@NumberTheorems{#1}}}}
  \NumberTheoremsIn{section}%
\def\NumberTheoremsAs#1{\@ifempty{#1}%
     {\cdr@thmsasfalse}%
     {\@ifundefined{c@#1}{\@nocounterr{#1}}%
       {\cdr@thmsinfalse\cdr@thmsastrue\cdr@thm@necountertrue\def\@NumberTheorems{#1}}}}
  \NumberTheoremsAs{}%
\def\SwapTheoremNumbers{\def\thm@swap{S}}
\def\NoSwapTheoremNumbers{\def\thm@swap{N}}
  \NoSwapTheoremNumbers
\def\cdr@thmdefs{%
  \theoremstyle{plain}
  \ifcdr@thm@necounter
  \ifcdr@thmsin
  \newcounter{cdrthm}[\@NumberTheorems]%
  \xdef\thecdrthm
         {\expandafter\noexpand\csname the\@NumberTheorems\endcsname
           \@thmcountersep\@thmcounter{cdrthm}}%
    \gdef\cdr@thm{cdrthm}%     
    \else
    \ifcdr@thmsas
    \xdef\cdr@thm{\@NumberTheorems}%
    \else
    \newcounter{cdrthm}%
    \gdef\cdr@thm{cdrthm}%     
    \fi\fi\fi
\ifcdr@thm@necounter
  \newtheorem{theo}[\cdr@thm]{\theoname}%
  \newtheorem{prop}[\cdr@thm]{\propname}%
  \newtheorem{conj}[\cdr@thm]{\conjname}%
  \newtheorem{coro}[\cdr@thm]{\coroname}%
  \newtheorem{lemm}[\cdr@thm]{\lemmname}%
  \theoremstyle{definition}%
  \newtheorem{defi}[\cdr@thm]{\definame}%
  \theoremstyle{remark}%
  \newtheorem{nota}[\cdr@thm]{\notaname}%
  \newtheorem{notas}[\cdr@thm]{\notasname}%
  \newtheorem{rema}[\cdr@thm]{\remaname}%
  \newtheorem{remas}[\cdr@thm]{\remasname}%
  \newtheorem{exem}[\cdr@thm]{\exemname}%
  \newtheorem{exems}[\cdr@thm]{\exemsname}%
\else
\ifcdr@thmsin
  \newtheorem{theo}{\theoname}[\@NumberTheorems]%
  \newtheorem{prop}{\propname}[\@NumberTheorems]%
  \newtheorem{conj}{\conjname}[\@NumberTheorems]%
  \newtheorem{coro}{\coroname}[\@NumberTheorems]%
  \newtheorem{lemm}{\lemmname}[\@NumberTheorems]%
  \theoremstyle{definition}%
  \newtheorem{defi}{\definame}[\@NumberTheorems]%
  \theoremstyle{remark}%
  \newtheorem{nota}{\notaname}[\@NumberTheorems]%
  \newtheorem{notas}{\notasname}[\@NumberTheorems]%
  \newtheorem{rema}{\remaname}[\@NumberTheorems]%
  \newtheorem{remas}{\remasname}[\@NumberTheorems]%
  \newtheorem{exem}{\exemname}[\@NumberTheorems]%
  \newtheorem{exems}{\exemsname}[\@NumberTheorems]%
\else
  \newtheorem{theo}{\theoname}%
  \newtheorem{prop}{\propname}%
  \newtheorem{conj}{\conjname}%
  \newtheorem{coro}{\coroname}%
  \newtheorem{lemm}{\lemmname}%
  \theoremstyle{definition}%
  \newtheorem{defi}{\definame}%
  \theoremstyle{remark}%
  \newtheorem{nota}{\notaname}%
  \newtheorem{notas}{\notasname}%
  \newtheorem{rema}{\remaname}%
  \newtheorem{remas}{\remasname}%
  \newtheorem{exem}{\exemname}%
  \newtheorem{exems}{\exemsname}%
\fi\fi
  \theoremstyle{definition}%
  \newtheorem*{defi*}{\definame}%
  \theoremstyle{remark}%
  \newtheorem*{nota*}{\notaname}%
  \newtheorem*{notas*}{\notasname}%
  \newtheorem*{rema*}{\remaname}%
  \newtheorem*{remas*}{\remasname}%
  \newtheorem*{exem*}{\exemname}%
  \newtheorem*{exems*}{\exemsname}%
  \theoremstyle{plain}%
  \newtheorem*{theo*}{\theoname}%
  \newtheorem*{prop*}{\propname}%
  \newtheorem*{conj*}{\conjname}%
  \newtheorem*{coro*}{\coroname}%
  \newtheorem*{lemm*}{\lemmname}%
}
\def\cdr@enoncedef{%
 \newenvironment{enonce}[2][plain]%
    {\let\cdrenonce\relax \theoremstyle{##1}%
      \ifcdr@thm@necounter
      \newtheorem{cdrenonce}[\cdr@thm]{##2}%
      \else
      \ifcdr@thmsin
      \newtheorem{cdrenonce}{##2}[\@NumberTheorems]%
      \else
      \newtheorem{cdrenonce}{##2}%
      \fi\fi
     \begin{cdrenonce}}%
    {\end{cdrenonce}}%
 \newenvironment{enonce*}[2][plain]%
    {\let\cdrenonce\relax \theoremstyle{##1}%
     \newtheorem*{cdrenonce}{##2}%
     \begin{cdrenonce}}%
    {\end{cdrenonce}}%
}
\AtBeginDocument{%
\cdr@thmdefs
\cdr@enoncedef}
\def\propname{Proposition}%
\def\conjname{Conjecture}%
\def\notaname{Notation}%
\def\notasname{Notations}%
\if@francais
\def\theoname{Th\'eor\`eme}%
\def\coroname{Corollaire}%
\def\lemmname{Lemme}%
\def\definame{D\'efinition}%
\def\remaname{Remarque}%
\def\remasname{Remarques}%
\def\exemname{Exemple}%
\def\exemsname{Exemples}%
\else
\def\theoname{Theorem}%
\def\coroname{Corollary}%
\def\lemmname{Lemma}%
\def\definame{Definition}%
\def\remaname{Remark}%
\def\remasname{Remarks}%
\def\exemname{Example}%
\def\exemsname{Examples}%
\fi
\def\equalenv#1#2{%
\newenvironment{#1}{\begin{#2}}{\end{#2}}}
\equalenv{theorem}{theo}
\equalenv{thm}{theo}
\equalenv{thm*}{theo*}
\equalenv{lemma}{lemm}
\equalenv{Remark}{rema}
\equalenv{Remarks}{remas}
\equalenv{corollary}{coro}
\equalenv{proposition}{prop}
\equalenv{definition}{defi}
\equalenv{remark}{rema}
\equalenv{remarks}{remas}
\equalenv{notation}{nota}
\equalenv{example}{exem}
\equalenv{Theoreme}{theo}
\equalenv{theoreme}{theo}
\equalenv{Lemme}{lemm}
\equalenv{lemme}{lemm}
\equalenv{Remarque}{rema}
\equalenv{remarque}{rema}
\equalenv{Remarques}{remas}
\equalenv{remarques}{remas}
\equalenv{Corollaire}{coro}
\equalenv{corollaire}{coro}
\equalenv{Proposition}{prop}
\equalenv{Definition}{defi}
\equalenv{Notation}{nota}
\equalenv{Exemple}{exem}
\equalenv{exemple}{exem}
%\equalenv{Proof}{proof}
%\equalenv{Preuve}{proof}

\newcommand{\QED}{\mbox{}\hfill \raisebox{-0.2pt}{\rule{5.6pt}{6pt}\rule{0pt}{0pt}} \medskip\par}

\newenvironment{Proof}{\noindent
    \parindent=0pt\abovedisplayskip = 0.5\abovedisplayskip
    \belowdisplayskip=\abovedisplayskip{\bfseries Proof. }}{\QED\hspace*{.5cm}}
 \newenvironment{ProofOf}[1]{\noindent
    \parindent=0pt\abovedisplayskip = 0.5\abovedisplayskip
    \belowdisplayskip=\abovedisplayskip{\bfseries Proof of  #1. }}{\QED\hspace*{.5cm}}

\let\proof\Proof
\let\endproof\endProof

%% Metadata
%  journal title
\newcommand{\setpublname}[2][Titre abr\'eg\'e absent]{%
\gdef\shortpublname{#1}\gdef\publname{#2}}
\setpublname{Journal of Computational Mathematics }
\def\currentjournaltitle{SMAI Journal of Computational Mathematics}
%  journal publisher
\edef\currentjournalpublisher{Soci\'et\'e de Math\'ematiques Appliqu\'ees et Industrielles}
% titre abr\'eg\'e du journal
\def\currentjournalshorttitle{SMAI-JCM}%
% acronym
\def\cdr@journalacro{SMAI-JCM}
% ISSN
\def\@issn{????-????}
%% article metadata
\let\@altabstract\@empty
\let\@alttitle\@empty
\newcommand*{\alttitle}[1]{\gdef\@alttitle{#1}}%
\newcommand*{\altkeywords}[1]{\gdef\@altkeywords{#1}}%
\newbox\altabstractbox
\newenvironment{altabstract}{%
  \ifx\maketitle\relax
    \ClassWarning{\@classname}{Altabstract should precede
      \protect\maketitle\space in AMS document classes; reported}%
  \fi
  \global\setbox\altabstractbox=\vtop \bgroup
    \normalfont\Small
    \list{}{\labelwidth\z@
      \leftmargin3pc \rightmargin\leftmargin
      \listparindent\normalparindent \itemindent\z@
      \parsep\z@ \@plus\p@
      \let\fullwidthdisplay\relax
    }%
    \item[\hskip\labelsep\bfseries\altabstractname.]%
}{%
  \endlist\egroup
  \ifx\@setaltabstract\relax \@setaltabstracta \fi
}
\def\@setaltabstract{\@setaltabstracta \global\let\@setaltabstract\relax}
\def\@setaltabstracta{%
\ifx\@empty\@alttitle\else
\begin{center}%
\large\itshape  \@alttitle  
\end{center}%
\fi
  \ifvoid\altabstractbox
  \else
%     \skip@20\p@ \advance\skip@-\lastskip
%     \advance\skip@-\baselineskip \vskip\skip@
    \box\altabstractbox
    \prevdepth\z@ % because \altabstractbox is a vtop
  \fi
}
\def\@setabstract{\@setabstracta \global\let\@setabstract\relax
\@setaltabstract
\ifx\@empty\@keywords\else \@setkeywords\fi
\ifx\@empty\@subjclass\else \@setsubjclass\fi
}
%% Abstracts
\def\abstractheadfont{\bfseries}
\def\alttitlefont{\large\itshape}
\def\@setabstracthead{\par{\abstractheadfont\abstractname\unskip.} \ignorespaces}%
\def\@setaltabstracthead{%
  \ifx\@alttitle\@empty\else
  \vskip16pt
  \begin{minipage}{\linewidth}
    \centering\alttitlefont\@alttitle
  \end{minipage}\fi
\nobreak\vskip8pt
\noindent
{\abstractheadfont\altabstractname\unskip.}%
\ignorespaces}%
%% copyright
\def\@setcopyright{}%
%%  page style
\def\ps@headings{ 
      \def\@oddhead{\hfil\textsc{\shorttitle}\hfil}%
      \def\@evenhead{\hfil\textsc{\shortauthors}\hfil}%
      \def\@oddfoot{\hfil\thepage\hfil}%
        \let\@evenfoot\@oddfoot
}
\pagestyle{headings}%
%% meant for A4 paper
\textwidth=170truemm
\textheight=223truemm
\ifx\undefined\pdfpagewidth\else
\pdfpagewidth\paperwidth
\pdfpageheight\paperheight
\fi
 \oddsidemargin=-5truemm
 \evensidemargin=-5truemm
 \headheight11pt
 \headsep22pt
 \footskip37pt
\frenchspacing
%% 11pt text
\def\@mainsize{11}\def\@ptsize{1}%
  \def\@typesizes{%
    \or{6}{7}\or{7}{8}\or{8}{10}\or{9}{11}\or{10}{12}%
    \or{\@xipt}{13}% normalsize
    \or{\@xiipt}{14}\or{\@xivpt}{17}\or{\@xviipt}{20}%
    \or{\@xxpt}{24}\or{\@xxvpt}{30}}%
  \normalsize \linespacing=\baselineskip
  \def\@listi{\leftmargin\leftmargini
            \parsep 5\p@  \@plus2.5\p@ \@minus\p@
            \topsep 10\p@ \@plus4\p@   \@minus6\p@
            \itemsep5\p@  \@plus2.5\p@ \@minus\p@
            \labelwidth=\leftmargini \advance\labelwidth-\labelsep }%
%
\let\@listI\@listi
\@listi
%% 1st page
\pagenumbering{arabic}%
%%  copyright
\def\@setcopyright{}%
%% First page style
\def\ps@firstpage{\ps@plain
      \def\@oddfoot{\hfil\thepage\hfil
\global\topskip\normaltopskip}%
        \let\@evenfoot\@oddfoot
  \def\@oddhead{\@serieslogo\hss}%
  \let\@evenhead\@oddhead % in case an article starts on a left-hand page
}
%
%% Identification article 
\def\@logofont{\small}
\def\article@logo{%
  \set@logo{%
	\vspace*{-8mm}\\
     \includegraphics[height=12mm]{logo_smai.jpg} \textit{\publname}\\
    \hspace*{15mm}\ifx\@empty\currentvolume \textbf{Version de travail}
    \else {Vol.~\currentvolume}, \pageinfo\ (\currentyear)%
    \fi
  }%
}
%
%% title
\def\@settitle{%
  \begin{center}%
    \vspace*{-28pt}
    \Large
    \bfseries
    \@title
  \end{center}%
}
%% authors
\def\@@and{\&}
\def\author@andify{%
  \nxandlist {\unskip ,\penalty-2 \space\ignorespaces}%
    {\unskip {} \@@and~}%
    {\unskip {} \penalty-1  \@@and~}%
}
\def\sjcm@author@andify{%
  \nxandlist{\unskip\\}{\unskip\\}{\unskip\\}}
\def\initial#1{#1}
\def\lastname#1{#1}
\def\firstname#1{#1}
\def\middlename#1{#1}
\def\vonname#1{#1}
\newcount\cdr@thankcount
\cdr@thankcount0
\def\@setauthors{%
  \begingroup
  \def\thanks{\protect\thanks@warning}%
  \trivlist
  \centering\footnotesize \@topsep22\p@\relax
  \advance\@topsep by -\baselineskip
  \item\relax
  \sjcm@author@andify\authors
  \def\\{\protect\linebreak}%
  \def\Same{\same}%
  \def\hasaddress##1##2{\def\What{##1}\ifx\What\Same\else\global\advance\cdr@thankcount1\fi\unskip
    \kern.2em\textsuperscript{\@arabic\cdr@thankcount}}%
  \large\textsc{\authors}%
  \ifx\@empty\contribs
  \else
    \\\penalty-3 \space \@setcontribs
    \@closetoccontribs
    \fi
    \linebreak
  \endtrivlist
  \endgroup
  \ifx\@empty\addresses \else
  \vglue-3mm
  \noindent\hspace*{3pc}\begin{minipage}[t]{13.3cm}
  \raggedright
  \normalfont\Small
  \@setaddresses
  \end{minipage}
  \vskip10pt plus 3pt minus 4pt\fi
}
\newcount\cdr@notes@adress\cdr@notes@adress0
\def\@setaddresses{%\par
   \begingroup
   \ifcdr@hyperref  \url@samestyle
   \else\def\url##1{##1}%
   \def\href##1##2{##2}\fi
%   \nobreak 
   % \interlinepenalty\@M
   \def\lastname##1{\relax}%
   \def\firstname##1{\relax}%
   \def\hasthanks##1{##1\@addpunct.\@par}
   \def\hasaddress##1{\relax}%
   \def\author##1{\ifnum\cdr@notes@adress=\z@\else\@@par\indent\fi\advance\cdr@notes@adress1
   \textsuperscript{\@arabic\cdr@notes@adress}\kern.2em\def\\{ \ignorespaces}\ignorespaces}%
%   \def\noauthor##1##2{}%
   \def\address##1##2{\begingroup
     {\ignorespaces##2\unskip}\endgroup}%
   \def\curraddr##1##2{\begingroup
     \@ifnotempty{##2}{\@@par\curraddrname{}: %
       {\ignorespaces##2\unskip}\endgroup}}%
   \def\email##1##2{\begingroup
     \@ifnotempty{##2}{\@@par\emailaddrname{}: %
       \href{mailto:##2}{\ttfamily ##2}}\endgroup}%
   \def\urladdr##1##2{\begingroup
%     \def~{\char`\~}%
     \@ifnotempty{##2}{\@@par\urladdrname{}: %
       \url{##2}}\endgroup}%
   \addresses\@addpunct.%
   \endgroup
 }
 %% Admin footnotes
\def\@adminfootnotes{%
  \let\@makefnmark\relax  \let\@thefnmark\relax
  \begingroup
    \parindent=0pt\normalparindent=0pt
    \ifx\@empty\thankses\else \@footnotetext{%
      \def\par{\let\par\@par}\@setthanks}%
    \fi
   \endgroup}
\def\@setthanks{\def\thanks##1{\par##1\@addpunct.}%
  \thankses}% 
\def\@setsubjclass{%
 \par\begingroup\footnotesize\list{}{\labelwidth\z@
      \leftmargin3pc \rightmargin\leftmargin
      \listparindent\normalparindent \itemindent\z@
      \parsep\z@ \@plus\p@
      \let\fullwidthdisplay\relax
    }%
    \item[\hskip\labelsep\bfseries \subjclassname.]\space \@subjclass\@addpunct.\par\endgroup\endlist}
\def\@setkeywords{%
  \par\begingroup\footnotesize\list{}{\labelwidth\z@
      \leftmargin3pc \rightmargin\leftmargin
      \listparindent\normalparindent \itemindent\z@
      \parsep\z@ \@plus\p@
      \let\fullwidthdisplay\relax
    }%
    \item[\hskip\labelsep\bfseries \keywordsname.]\space \@keywords\@addpunct.\par\endgroup\endlist}
\if@francais
 \def\keywordsname{Mots-cl\'es}
 \def\subjclassname{Classification math.}
 \def\abstractname{\textbf{R\'esum\'e}}
 \def\altabstractname{\textbf{Abstract}}
\else
 \def\keywordsname{Keywords}
 \def\subjclassname{Math. classification}
 \def\abstractname{\bf{Abstract}}
 \def\altabstractname{\textbf{R\'esum\'e}}
\fi
\@xp\let\csname subjclassname@2000\endcsname \subjclassname
\@xp\let\csname subjclassname@1991\endcsname \subjclassname
%% Last page
\newcount\auth@count
\auth@count0
\AtBeginDocument{\@ifpackageloaded{hyperref}{}{\def\url#1{#1}%
  \def\href#1#2{#2}}}
%% final
\def\enddoc@text{}%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% sections
\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                                   {\normalfont\large\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\normalsize\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\normalfont\normalsize\itshape}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                    {3.25ex \@plus1ex \@minus.2ex}%
                                    {-1em}%
                                    {\normalfont\normalsize\bfseries}}
\renewcommand\subparagraph{\@startsection{subparagraph}{4}{\parindent}%
                                       {3.25ex \@plus1ex \@minus .2ex}%
                                       {-1em}%
                                      {\normalfont\normalsize\itshape}}
\let\paragraphc\paragraph
\let\subparagraphc\subparagraph
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}
\numberwithin{equation}{section}
%% Biblio
\renewcommand{\@bibtitlestyle}{%
  \section*{\bibname}%
}
\let\bibliofont\normalfont
%% Theorems
\OneNumberAllTheorems
\def\th@definition{%
  \thm@headfont{\bfseries}%
  \normalfont % body font
}
\def\th@remark{%
  \thm@headfont{\bfseries}%
  \normalfont % body font
}
\endinput
