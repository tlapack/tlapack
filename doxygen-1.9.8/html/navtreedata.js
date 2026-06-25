/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "Doxygen", "index.html", [
    [ "Installation", "install.html", [
      [ "Compiling from source on UNIX", "install.html#install_src_unix", null ],
      [ "Installing the binaries on UNIX", "install.html#install_bin_unix", null ],
      [ "Compiling from source on Windows", "install.html#install_src_windows", null ],
      [ "Installing the binaries on Windows", "install.html#install_bin_windows", null ]
    ] ],
    [ "Getting started", "starting.html", [
      [ "Step 0: Check if doxygen supports your programming/hardware description language", "starting.html#step0", null ],
      [ "Step 1: Creating a configuration file", "starting.html#step1", null ],
      [ "Step 2: Running doxygen", "starting.html#step2", [
        [ "HTML output", "starting.html#html_out", null ],
        [ "LaTeX output", "starting.html#latex_out", null ],
        [ "RTF output", "starting.html#rtf_out", null ],
        [ "XML output", "starting.html#xml_out", null ],
        [ "Man page output", "starting.html#man_out", null ],
        [ "DocBook output", "starting.html#docbook_out", null ]
      ] ],
      [ "Step 3: Documenting the sources", "starting.html#step3", null ]
    ] ],
    [ "Documenting the code", "docblocks.html", [
      [ "Special comment blocks", "docblocks.html#specialblock", [
        [ "Comment blocks for C-like languages (C/C++/C#/Objective-C/PHP/Java)", "docblocks.html#cppblock", [
          [ "Putting documentation after members", "docblocks.html#memberdoc", null ],
          [ "Examples", "docblocks.html#docexamples", null ],
          [ "Documentation at other places", "docblocks.html#structuralcommands", null ]
        ] ],
        [ "Comment blocks in Python", "docblocks.html#pythonblocks", null ],
        [ "Comment blocks in VHDL", "docblocks.html#vhdlblocks", null ],
        [ "Comment blocks in Fortran", "docblocks.html#fortranblocks", null ]
      ] ],
      [ "Anatomy of a comment block", "docblocks.html#docstructure", null ]
    ] ],
    [ "Markdown support", "markdown.html", [
      [ "Standard Markdown", "markdown.html#markdown_std", [
        [ "Paragraphs", "markdown.html#md_para", null ],
        [ "Headers", "markdown.html#md_headers", null ],
        [ "Block quotes", "markdown.html#md_blockquotes", null ],
        [ "Lists", "markdown.html#md_lists", null ],
        [ "Code Blocks", "markdown.html#md_codeblock", null ],
        [ "Horizontal Rulers", "markdown.html#md_rulers", null ],
        [ "Emphasis", "markdown.html#md_emphasis", null ],
        [ "Strikethrough", "markdown.html#md_strikethrough", null ],
        [ "code spans", "markdown.html#md_codespan", null ],
        [ "Links", "markdown.html#md_links", [
          [ "Inline Links", "markdown.html#md_inlinelinks", null ],
          [ "Reference Links", "markdown.html#md_reflinks", null ]
        ] ],
        [ "Images", "markdown.html#md_images", null ],
        [ "Automatic Linking", "markdown.html#md_autolink", null ]
      ] ],
      [ "Markdown Extensions", "markdown.html#markdown_extra", [
        [ "Table of Contents", "markdown.html#md_toc", null ],
        [ "Tables", "markdown.html#md_tables", null ],
        [ "Fenced Code Blocks", "markdown.html#md_fenced", null ],
        [ "Header Id Attributes", "markdown.html#md_header_id", null ],
        [ "Image Attributes", "markdown.html#md_image_attributes", null ]
      ] ],
      [ "Doxygen specifics", "markdown.html#markdown_dox", [
        [ "Including Markdown files as pages", "markdown.html#md_page_header", null ],
        [ "Treatment of HTML blocks", "markdown.html#md_html_blocks", null ],
        [ "Code Block Indentation", "markdown.html#mddox_code_blocks", null ],
        [ "Emphasis and strikethrough limits", "markdown.html#mddox_emph_spans", null ],
        [ "Code Spans Limits", "markdown.html#mddox_code_spans", null ],
        [ "Lists Extensions", "markdown.html#mddox_lists", null ],
        [ "Use of asterisks", "markdown.html#mddox_stars", null ],
        [ "Limits on markup scope", "markdown.html#mddox_limits", null ]
      ] ],
      [ "Debugging problems", "markdown.html#markdown_debug", null ]
    ] ],
    [ "Lists", "lists.html", null ],
    [ "Grouping", "grouping.html", [
      [ "Topics", "grouping.html#topics", null ],
      [ "Member Groups", "grouping.html#memgroup", null ],
      [ "Subpaging", "grouping.html#subpaging", null ]
    ] ],
    [ "Including formulas", "formulas.html", null ],
    [ "Including tables", "tables.html", null ],
    [ "Graphs and diagrams", "diagrams.html", null ],
    [ "Preprocessing", "preprocessing.html", null ],
    [ "Automatic link generation", "autolink.html", [
      [ "Links to web pages and mail addresses", "autolink.html#linkurl", null ],
      [ "Links to classes", "autolink.html#linkclass", null ],
      [ "Links to files", "autolink.html#linkfile", null ],
      [ "Links to functions", "autolink.html#linkfunc", null ],
      [ "Links to other members", "autolink.html#linkother", null ],
      [ "typedefs", "autolink.html#resolving", null ]
    ] ],
    [ "Output Formats", "output.html", null ],
    [ "Searching", "searching.html", [
      [ "External Indexing and Searching", "extsearch.html", [
        [ "Introduction", "extsearch.html#extsearch_intro", null ],
        [ "Configuring", "extsearch.html#extsearch_config", [
          [ "Single project index", "extsearch.html#extsearch_single", null ],
          [ "Multi project index", "extsearch.html#extsearch_multi", null ]
        ] ],
        [ "Updating the index", "extsearch.html#extsearch_update", null ],
        [ "Programming interface", "extsearch.html#extsearch_api", [
          [ "Indexer input format", "extsearch.html#extsearch_api_index", null ],
          [ "Search URL format", "extsearch.html#extsearch_api_search_in", null ],
          [ "Search results format", "extsearch.html#extsearch_api_search_out", null ]
        ] ]
      ] ]
    ] ],
    [ "Customizing the output", "customize.html", [
      [ "Minor Tweaks", "customize.html#minor_tweaks", [
        [ "Overall Color", "customize.html#minor_tweaks_colors", null ],
        [ "Navigation", "customize.html#minor_tweaks_treeview", null ],
        [ "Dynamic Content", "customize.html#minor_tweaks_dynsection", null ],
        [ "Header, Footer, and Stylesheet changes", "customize.html#minor_tweaks_header_css", null ]
      ] ],
      [ "Changing the layout of pages", "customize.html#layout", null ],
      [ "Using the XML output", "customize.html#xmlgenerator", null ]
    ] ],
    [ "Custom Commands", "custcmd.html", [
      [ "Simple aliases", "custcmd.html#custcmd_simple", null ],
      [ "Aliases with arguments", "custcmd.html#custcmd_complex", null ],
      [ "Nesting custom command", "custcmd.html#custcmd_nesting", null ]
    ] ],
    [ "Linking to external documentation", "external.html", null ],
    [ "Frequently Asked Questions", "faq.html", [
      [ "How to get information on the index page in HTML?", "faq.html#faq_index", null ],
      [ "Help, some/all of the members of my class / file / namespace are not documented?", "faq.html#fac_al", null ],
      [ "When I set EXTRACT_ALL to NO none of my functions are shown in the documentation.", "faq.html#faq_extract_all", null ],
      [ "My file with a custom extension is not parsed (properly) (anymore).", "faq.html#faq_ext_mapping", null ],
      [ "How can I make doxygen ignore some code fragment?", "faq.html#faq_code", null ],
      [ "How can I change what is after the #include in the class documentation?", "faq.html#faq_code_inc", null ],
      [ "How can I use tag files in combination with compressed HTML?", "faq.html#faq_chm", null ],
      [ "I don't like the quick index that is put above each HTML page, what do I do?", "faq.html#faq_html", null ],
      [ "The overall HTML output looks different, while I only wanted to use my own html header file", "faq.html#faq_html_header", null ],
      [ "Why does doxygen use Qt?", "faq.html#faq_use_qt", null ],
      [ "How can I exclude all test directories from my directory tree?", "faq.html#faq_excl_dir", null ],
      [ "Doxygen automatically generates a link to the class MyClass somewhere in the running text.  How do I prevent that at a certain place?", "faq.html#faq_class", null ],
      [ "My favorite programming language is X. Can I still use doxygen?", "faq.html#faq_pgm_X", null ],
      [ "Help! I get the cryptic message \"input buffer overflow, can't enlarge buffer because scanner uses REJECT\"", "faq.html#faq_lex", null ],
      [ "When running make in the latex directory I get \"TeX capacity exceeded\". Now what?", "faq.html#faq_latex", null ],
      [ "Why are dependencies via STL classes not shown in the dot graphs?", "faq.html#faq_stl", null ],
      [ "I have problems getting the search engine to work with PHP5 and/or windows", "faq.html#faq_search", null ],
      [ "Can I configure doxygen from the command line?", "faq.html#faq_cmdline", null ],
      [ "How did doxygen get its name?", "faq.html#faq_name", null ],
      [ "What was the reason to develop doxygen?", "faq.html#faq_why", null ],
      [ "How to prevent interleaved output", "faq.html#faq_bin", null ]
    ] ],
    [ "Troubleshooting", "trouble.html", [
      [ "Known Problems", "trouble.html#knowproblems", null ],
      [ "How to Help", "trouble.html#howtohelp", null ],
      [ "How to report a bug", "trouble.html#bug_reports", null ]
    ] ],
    [ "Features", "features.html", null ],
    [ "Doxygen usage", "doxygen_usage.html", [
      [ "Fine-tuning the output", "doxygen_usage.html#doxygen_finetune", null ]
    ] ],
    [ "Doxywizard usage", "doxywizard_usage.html", [
      [ "Wizard tab", "doxywizard_usage.html#dw_wizard", [
        [ "Project settings", "doxywizard_usage.html#dw_wizard_project", null ],
        [ "Mode of operating", "doxywizard_usage.html#dw_wizard_mode", null ],
        [ "Output to produce", "doxywizard_usage.html#dw_wizard_output", null ],
        [ "Diagrams to generate", "doxywizard_usage.html#dw_wizard_diagrams", null ]
      ] ],
      [ "Expert tab", "doxywizard_usage.html#dw_expert", null ],
      [ "Run tab", "doxywizard_usage.html#dw_run", null ],
      [ "Menu options", "doxywizard_usage.html#dw_menu", [
        [ "File menu", "doxywizard_usage.html#dw_menu_file", null ],
        [ "Settings menu", "doxywizard_usage.html#dw_menu_settings", null ],
        [ "Help menu", "doxywizard_usage.html#dw_menu_help", null ]
      ] ]
    ] ],
    [ "Configuration", "config.html", [
      [ "Format", "config.html#config_format", null ],
      [ "Project related configuration options", "config.html#config_project", null ],
      [ "Build related configuration options", "config.html#config_build", null ],
      [ "Configuration options related to warning and progress messages", "config.html#config_messages", null ],
      [ "Configuration options related to the input files", "config.html#config_input", null ],
      [ "Configuration options related to source browsing", "config.html#config_source_browser", null ],
      [ "Configuration options related to the alphabetical class index", "config.html#config_index", null ],
      [ "Configuration options related to the HTML output", "config.html#config_html", null ],
      [ "Configuration options related to the LaTeX output", "config.html#config_latex", null ],
      [ "Configuration options related to the RTF output", "config.html#config_rtf", null ],
      [ "Configuration options related to the man page output", "config.html#config_man", null ],
      [ "Configuration options related to the XML output", "config.html#config_xml", null ],
      [ "Configuration options related to the DOCBOOK output", "config.html#config_docbook", null ],
      [ "Configuration options for the AutoGen Definitions output", "config.html#config_autogen", null ],
      [ "Configuration options related to Sqlite3 output", "config.html#config_sqlite3", null ],
      [ "Configuration options related to the Perl module output", "config.html#config_perlmod", null ],
      [ "Configuration options related to the preprocessor", "config.html#config_preprocessor", null ],
      [ "Configuration options related to external references", "config.html#config_external", null ],
      [ "Configuration options related to diagram generator tools", "config.html#config_dot", null ],
      [ "Examples", "config.html#config_examples", null ]
    ] ],
    [ "Special Commands", "commands.html", [
      [ "Introduction", "commands.html#cmd_intro", null ],
      [ "\\addtogroup <name> [(title)]", "commands.html#cmdaddtogroup", null ],
      [ "\\callgraph", "commands.html#cmdcallgraph", null ],
      [ "\\hidecallgraph", "commands.html#cmdhidecallgraph", null ],
      [ "\\callergraph", "commands.html#cmdcallergraph", null ],
      [ "\\hidecallergraph", "commands.html#cmdhidecallergraph", null ],
      [ "\\showrefby", "commands.html#cmdshowrefby", null ],
      [ "\\hiderefby", "commands.html#cmdhiderefby", null ],
      [ "\\showrefs", "commands.html#cmdshowrefs", null ],
      [ "\\hiderefs", "commands.html#cmdhiderefs", null ],
      [ "\\includegraph", "commands.html#cmdincludegraph", null ],
      [ "\\hideincludegraph", "commands.html#cmdhideincludegraph", null ],
      [ "\\includedbygraph", "commands.html#cmdincludedbygraph", null ],
      [ "\\hideincludedbygraph", "commands.html#cmdhideincludedbygraph", null ],
      [ "\\directorygraph", "commands.html#cmddirectorygraph", null ],
      [ "\\hidedirectorygraph", "commands.html#cmdhidedirectorygraph", null ],
      [ "\\collaborationgraph", "commands.html#cmdcollaborationgraph", null ],
      [ "\\hidecollaborationgraph", "commands.html#cmdhidecollaborationgraph", null ],
      [ "\\groupgraph", "commands.html#cmdgroupgraph", null ],
      [ "\\hidegroupgraph", "commands.html#cmdhidegroupgraph", null ],
      [ "\\qualifier <label> | \"(text)\"", "commands.html#cmdqualifier", null ],
      [ "\\category <name> [<header-file>] [<header-name>]", "commands.html#cmdcategory", null ],
      [ "\\class <name> [<header-file>] [<header-name>]", "commands.html#cmdclass", null ],
      [ "\\concept <name>", "commands.html#cmdconcept", null ],
      [ "\\def <name>", "commands.html#cmddef", null ],
      [ "\\defgroup <name> (group title)", "commands.html#cmddefgroup", null ],
      [ "\\dir [<path fragment>]", "commands.html#cmddir", null ],
      [ "\\enum <name>", "commands.html#cmdenum", null ],
      [ "\\example['{lineno}'] <file-name>", "commands.html#cmdexample", null ],
      [ "\\endinternal", "commands.html#cmdendinternal", null ],
      [ "\\extends <name>", "commands.html#cmdextends", null ],
      [ "\\file [<name>]", "commands.html#cmdfile", null ],
      [ "\\fileinfo['{'option'}']", "commands.html#cmdfileinfo", null ],
      [ "\\lineinfo", "commands.html#cmdlineinfo", null ],
      [ "\\fn (function declaration)", "commands.html#cmdfn", null ],
      [ "\\headerfile <header-file> [<header-name>]", "commands.html#cmdheaderfile", null ],
      [ "\\hideinitializer", "commands.html#cmdhideinitializer", null ],
      [ "\\idlexcept <name>", "commands.html#cmdidlexcept", null ],
      [ "\\implements <name>", "commands.html#cmdimplements", null ],
      [ "\\ingroup (<groupname> [<groupname>]*)", "commands.html#cmdingroup", null ],
      [ "\\interface <name> [<header-file>] [<header-name>]", "commands.html#cmdinterface", null ],
      [ "\\internal", "commands.html#cmdinternal", null ],
      [ "\\mainpage [(title)]", "commands.html#cmdmainpage", null ],
      [ "\\memberof <name>", "commands.html#cmdmemberof", null ],
      [ "\\module <name>", "commands.html#cmdmodule", null ],
      [ "\\name [(header)]", "commands.html#cmdname", null ],
      [ "\\namespace <name>", "commands.html#cmdnamespace", null ],
      [ "\\nosubgrouping", "commands.html#cmdnosubgrouping", null ],
      [ "\\overload [(function declaration)]", "commands.html#cmdoverload", null ],
      [ "\\package <name>", "commands.html#cmdpackage", null ],
      [ "\\page <name> (title)", "commands.html#cmdpage", null ],
      [ "\\private", "commands.html#cmdprivate", null ],
      [ "\\privatesection", "commands.html#cmdprivatesection", null ],
      [ "\\property (qualified property name)", "commands.html#cmdproperty", null ],
      [ "\\protected", "commands.html#cmdprotected", null ],
      [ "\\protectedsection", "commands.html#cmdprotectedsection", null ],
      [ "\\protocol <name> [<header-file>] [<header-name>]", "commands.html#cmdprotocol", null ],
      [ "\\public", "commands.html#cmdpublic", null ],
      [ "\\publicsection", "commands.html#cmdpublicsection", null ],
      [ "\\pure", "commands.html#cmdpure", null ],
      [ "\\relates <name>", "commands.html#cmdrelates", null ],
      [ "\\related <name>", "commands.html#cmdrelated", null ],
      [ "\\relatesalso <name>", "commands.html#cmdrelatesalso", null ],
      [ "\\relatedalso <name>", "commands.html#cmdrelatedalso", null ],
      [ "\\showinitializer", "commands.html#cmdshowinitializer", null ],
      [ "\\static", "commands.html#cmdstatic", null ],
      [ "\\struct <name> [<header-file>] [<header-name>]", "commands.html#cmdstruct", null ],
      [ "\\typedef (typedef declaration)", "commands.html#cmdtypedef", null ],
      [ "\\union <name> [<header-file>] [<header-name>]", "commands.html#cmdunion", null ],
      [ "\\var (variable declaration)", "commands.html#cmdvar", null ],
      [ "\\vhdlflow [(title for the flow chart)]", "commands.html#cmdvhdlflow", null ],
      [ "\\weakgroup <name> [(title)]", "commands.html#cmdweakgroup", null ],
      [ "\\attention { attention text }", "commands.html#cmdattention", null ],
      [ "\\author { list of authors }", "commands.html#cmdauthor", null ],
      [ "\\authors { list of authors }", "commands.html#cmdauthors", null ],
      [ "\\brief { brief description }", "commands.html#cmdbrief", null ],
      [ "\\bug { bug description }", "commands.html#cmdbug", null ],
      [ "\\cond [(section-label)]", "commands.html#cmdcond", null ],
      [ "\\copyright { copyright description }", "commands.html#cmdcopyright", null ],
      [ "\\date { date description }", "commands.html#cmddate", null ],
      [ "\\showdate \"<format>\" [ <date_time> ]", "commands.html#cmdshowdate", null ],
      [ "\\deprecated { description }", "commands.html#cmddeprecated", null ],
      [ "\\details { detailed description }", "commands.html#cmddetails", null ],
      [ "\\noop ( text to be ignored )", "commands.html#cmdnoop", null ],
      [ "\\raisewarning ( text to be shown as warning )", "commands.html#cmdraisewarning", null ],
      [ "\\else", "commands.html#cmdelse", null ],
      [ "\\elseif (section-label)", "commands.html#cmdelseif", null ],
      [ "\\endcond", "commands.html#cmdendcond", null ],
      [ "\\endif", "commands.html#cmdendif", null ],
      [ "\\exception <exception-object> { exception description }", "commands.html#cmdexception", null ],
      [ "\\if (section-label)", "commands.html#cmdif", null ],
      [ "\\ifnot (section-label)", "commands.html#cmdifnot", null ],
      [ "\\invariant { description of invariant }", "commands.html#cmdinvariant", null ],
      [ "\\note { text }", "commands.html#cmdnote", null ],
      [ "\\par [(paragraph title)] { paragraph }", "commands.html#cmdpar", null ],
      [ "\\param '['dir']' <parameter-name> { parameter description }", "commands.html#cmdparam", null ],
      [ "\\parblock", "commands.html#cmdparblock", null ],
      [ "\\endparblock", "commands.html#cmdendparblock", null ],
      [ "\\tparam <template-parameter-name> { description }", "commands.html#cmdtparam", null ],
      [ "\\post { description of the postcondition }", "commands.html#cmdpost", null ],
      [ "\\pre { description of the precondition }", "commands.html#cmdpre", null ],
      [ "\\remark { remark text }", "commands.html#cmdremark", null ],
      [ "\\remarks { remark text }", "commands.html#cmdremarks", null ],
      [ "\\result { description of the result value }", "commands.html#cmdresult", null ],
      [ "\\return { description of the return value }", "commands.html#cmdreturn", null ],
      [ "\\returns { description of the return value }", "commands.html#cmdreturns", null ],
      [ "\\retval <return value> { description }", "commands.html#cmdretval", null ],
      [ "\\sa { references }", "commands.html#cmdsa", null ],
      [ "\\see { references }", "commands.html#cmdsee", null ],
      [ "\\short { short description }", "commands.html#cmdshort", null ],
      [ "\\since { text }", "commands.html#cmdsince", null ],
      [ "\\test { paragraph describing a test case }", "commands.html#cmdtest", null ],
      [ "\\throw <exception-object> { exception description }", "commands.html#cmdthrow", null ],
      [ "\\throws <exception-object> { exception description }", "commands.html#cmdthrows", null ],
      [ "\\todo { paragraph describing what is to be done }", "commands.html#cmdtodo", null ],
      [ "\\version { version number }", "commands.html#cmdversion", null ],
      [ "\\warning { warning message }", "commands.html#cmdwarning", null ],
      [ "\\xrefitem <key> \"heading\" \"list title\" { text }", "commands.html#cmdxrefitem", null ],
      [ "\\addindex (text)", "commands.html#cmdaddindex", null ],
      [ "\\anchor <word>", "commands.html#cmdanchor", null ],
      [ "\\cite <label>", "commands.html#cmdcite", null ],
      [ "\\endlink", "commands.html#cmdendlink", null ],
      [ "\\link <link-object>", "commands.html#cmdlink", null ],
      [ "\\ref <name> [\"(text)\"]", "commands.html#cmdref", null ],
      [ "\\refitem <name>", "commands.html#cmdrefitem", null ],
      [ "\\secreflist", "commands.html#cmdsecreflist", null ],
      [ "\\endsecreflist", "commands.html#cmdendsecreflist", null ],
      [ "\\subpage <name> [\"(text)\"]", "commands.html#cmdsubpage", null ],
      [ "\\tableofcontents['{'[option[:level]][,option[:level]]*'}']", "commands.html#cmdtableofcontents", null ],
      [ "\\section <section-name> (section title)", "commands.html#cmdsection", null ],
      [ "\\subsection <subsection-name> (subsection title)", "commands.html#cmdsubsection", null ],
      [ "\\subsubsection <subsubsection-name> (subsubsection title)", "commands.html#cmdsubsubsection", null ],
      [ "\\paragraph <paragraph-name> (paragraph title)", "commands.html#cmdparagraph", null ],
      [ "\\dontinclude['{lineno}'] <file-name>", "commands.html#cmddontinclude", null ],
      [ "\\include['{'option'}'] <file-name>", "commands.html#cmdinclude", null ],
      [ "\\includelineno <file-name>", "commands.html#cmdincludelineno", null ],
      [ "\\includedoc <file-name>", "commands.html#cmdincludedoc", null ],
      [ "\\line ( pattern )", "commands.html#cmdline", null ],
      [ "\\skip ( pattern )", "commands.html#cmdskip", null ],
      [ "\\skipline ( pattern )", "commands.html#cmdskipline", null ],
      [ "\\snippet['{'option'}'] <file-name> ( block_id )", "commands.html#cmdsnippet", null ],
      [ "\\snippetlineno <file-name> ( block_id )", "commands.html#cmdsnippetlineno", null ],
      [ "\\snippetdoc <file-name> ( block_id )", "commands.html#cmdsnippetdoc", null ],
      [ "\\until ( pattern )", "commands.html#cmduntil", null ],
      [ "\\verbinclude <file-name>", "commands.html#cmdverbinclude", null ],
      [ "\\htmlinclude [\"[block]\"] <file-name>", "commands.html#cmdhtmlinclude", null ],
      [ "\\latexinclude <file-name>", "commands.html#cmdlatexinclude", null ],
      [ "\\rtfinclude <file-name>", "commands.html#cmdrtfinclude", null ],
      [ "\\maninclude <file-name>", "commands.html#cmdmaninclude", null ],
      [ "\\docbookinclude <file-name>", "commands.html#cmddocbookinclude", null ],
      [ "\\xmlinclude <file-name>", "commands.html#cmdxmlinclude", null ],
      [ "\\a <word>", "commands.html#cmda", null ],
      [ "\\arg { item-description }", "commands.html#cmdarg", null ],
      [ "\\b <word>", "commands.html#cmdb", null ],
      [ "\\c <word>", "commands.html#cmdc", null ],
      [ "\\code['{'<word>'}']", "commands.html#cmdcode", null ],
      [ "\\copydoc <link-object>", "commands.html#cmdcopydoc", null ],
      [ "\\copybrief <link-object>", "commands.html#cmdcopybrief", null ],
      [ "\\copydetails <link-object>", "commands.html#cmdcopydetails", null ],
      [ "\\docbookonly", "commands.html#cmddocbookonly", null ],
      [ "\\dot [\"caption\"] [<sizeindication>=<size>]", "commands.html#cmddot", null ],
      [ "\\emoji \"name\"", "commands.html#cmdemoji", null ],
      [ "\\msc [\"caption\"] [<sizeindication>=<size>]", "commands.html#cmdmsc", null ],
      [ "\\startuml ['{'option[,option]'}'] [\"caption\"] [<sizeindication>=<size>]", "commands.html#cmdstartuml", null ],
      [ "\\dotfile <file> [\"caption\"] [<sizeindication>=<size>]", "commands.html#cmddotfile", null ],
      [ "\\mscfile <file> [\"caption\"] [<sizeindication>=<size>]", "commands.html#cmdmscfile", null ],
      [ "\\diafile <file> [\"caption\"] [<sizeindication>=<size>]", "commands.html#cmddiafile", null ],
      [ "\\doxyconfig <config_option>", "commands.html#cmddoxyconfig", null ],
      [ "\\e <word>", "commands.html#cmde", null ],
      [ "\\em <word>", "commands.html#cmdem", null ],
      [ "\\endcode", "commands.html#cmdendcode", null ],
      [ "\\enddocbookonly", "commands.html#cmdenddocbookonly", null ],
      [ "\\enddot", "commands.html#cmdenddot", null ],
      [ "\\endmsc", "commands.html#cmdendmsc", null ],
      [ "\\enduml", "commands.html#cmdenduml", null ],
      [ "\\endhtmlonly", "commands.html#cmdendhtmlonly", null ],
      [ "\\endlatexonly", "commands.html#cmdendlatexonly", null ],
      [ "\\endmanonly", "commands.html#cmdendmanonly", null ],
      [ "\\endrtfonly", "commands.html#cmdendrtfonly", null ],
      [ "\\endverbatim", "commands.html#cmdendverbatim", null ],
      [ "\\endxmlonly", "commands.html#cmdendxmlonly", null ],
      [ "\\f$", "commands.html#cmdfdollar", null ],
      [ "\\f(", "commands.html#cmdfrndopen", null ],
      [ "\\f)", "commands.html#cmdfrndclose", null ],
      [ "\\f[", "commands.html#cmdfbropen", null ],
      [ "\\f]", "commands.html#cmdfbrclose", null ],
      [ "\\f{environment}{", "commands.html#cmdfcurlyopen", null ],
      [ "\\f}", "commands.html#cmdfcurlyclose", null ],
      [ "\\htmlonly [\"[block]\"]", "commands.html#cmdhtmlonly", null ],
      [ "\\image['{'option[,option]'}'] <format> <file> [\"caption\"] [<sizeindication>=<size>]", "commands.html#cmdimage", null ],
      [ "\\latexonly", "commands.html#cmdlatexonly", null ],
      [ "\\manonly", "commands.html#cmdmanonly", null ],
      [ "\\li { item-description }", "commands.html#cmdli", null ],
      [ "\\n", "commands.html#cmdn", null ],
      [ "\\p <word>", "commands.html#cmdp", null ],
      [ "\\rtfonly", "commands.html#cmdrtfonly", null ],
      [ "\\verbatim", "commands.html#cmdverbatim", null ],
      [ "\\xmlonly", "commands.html#cmdxmlonly", null ],
      [ "\\\\", "commands.html#cmdbackslash", null ],
      [ "\\@", "commands.html#cmdat", null ],
      [ "\\~[LanguageId]", "commands.html#cmdtilde", null ],
      [ "\\&", "commands.html#cmdamp", null ],
      [ "\\$", "commands.html#cmddollar", null ],
      [ "\\#", "commands.html#cmdhash", null ],
      [ "\\<", "commands.html#cmdlt", null ],
      [ "\\>", "commands.html#cmdgt", null ],
      [ "\\%", "commands.html#cmdperc", null ],
      [ "\\\"", "commands.html#cmdquot", null ],
      [ "\\.", "commands.html#cmdchardot", null ],
      [ "\\=", "commands.html#cmdeq", null ],
      [ "\\::", "commands.html#cmddcolon", null ],
      [ "\\|", "commands.html#cmdpipe", null ],
      [ "\\--", "commands.html#cmdndash", null ],
      [ "\\---", "commands.html#cmdmdash", null ]
    ] ],
    [ "HTML Commands", "htmlcmds.html", [
      [ "HTML tag commands", "htmlcmds.html#htmltagcmds", null ],
      [ "HTML4 character entities", "htmlcmds.html#htmlentities", null ]
    ] ],
    [ "XML Commands", "xmlcmds.html", null ],
    [ "Emoji support", "emojisup.html", [
      [ "Representation", "emojisup.html#emojirep", null ],
      [ "Emoji image retrieval", "emojisup.html#emojiimage", null ]
    ] ],
    [ "Internationalization", "langhowto.html", null ],
    [ "Perl Module Output", "perlmod.html", [
      [ "Usage", "perlmod.html#using_perlmod_fmt", null ],
      [ "Using the LaTeX generator.", "perlmod.html#perlmod_latex", [
        [ "Creation of PDF and DVI output", "perlmod.html#pm_pdf_gen", null ]
      ] ],
      [ "Documentation format.", "perlmod.html#doxydocs_format", null ],
      [ "Data structure", "perlmod.html#doxymodel_format", null ],
      [ "Perl Module Tree Nodes", "perlmod.html#perlmod_tree", null ]
    ] ],
    [ "Doxygen's Internals", "arch.html", null ],
    [ "Changelog", "changelog.html", [
      [ "1.9 Series", "changelog.html#log_1_9", [
        [ "Release 1.9.8", "changelog.html#log_1_9_8", null ],
        [ "Release 1.9.7", "changelog.html#log_1_9_7", null ],
        [ "Release 1.9.6", "changelog.html#log_1_9_6", null ],
        [ "Release 1.9.5", "changelog.html#log_1_9_5", null ],
        [ "Release 1.9.4", "changelog.html#log_1_9_4", null ],
        [ "Release 1.9.3", "changelog.html#log_1_9_3", null ],
        [ "Release 1.9.2", "changelog.html#log_1_9_2", null ],
        [ "Release 1.9.1", "changelog.html#log_1_9_1", null ],
        [ "Release 1.9.0", "changelog.html#log_1_9_0", null ]
      ] ],
      [ "1.8 Series", "changelog.html#log_1_8", [
        [ "Release 1.8.20", "changelog.html#log_1_8_20", null ],
        [ "Release 1.8.19", "changelog.html#log_1_8_19", null ],
        [ "Release 1.8.18", "changelog.html#log_1_8_18", null ],
        [ "Release 1.8.17", "changelog.html#log_1_8_17", null ],
        [ "Release 1.8.16", "changelog.html#log_1_8_16", null ],
        [ "Release 1.8.15", "changelog.html#log_1_8_15", null ],
        [ "Release 1.8.14", "changelog.html#log_1_8_14", null ],
        [ "Release 1.8.13", "changelog.html#log_1_8_13", null ],
        [ "Release 1.8.12", "changelog.html#log_1_8_12", null ],
        [ "Release 1.8.11", "changelog.html#log_1_8_11", null ],
        [ "Release 1.8.10", "changelog.html#log_1_8_10", null ],
        [ "Release 1.8.9.1", "changelog.html#log_1_8_9_1", null ],
        [ "Release 1.8.9", "changelog.html#log_1_8_9", null ],
        [ "Release 1.8.8", "changelog.html#log_1_8_8", null ],
        [ "Release 1.8.7", "changelog.html#log_1_8_7", null ],
        [ "Release 1.8.6", "changelog.html#log_1_8_6", null ],
        [ "Release 1.8.5", "changelog.html#log_1_8_5", null ],
        [ "Release 1.8.4", "changelog.html#log_1_8_4", null ],
        [ "Release 1.8.3.1", "changelog.html#log_1_8_3_1", null ],
        [ "Release 1.8.3", "changelog.html#log_1_8_3", null ],
        [ "Release 1.8.2", "changelog.html#log_1_8_2", null ],
        [ "Release 1.8.1.2", "changelog.html#log_1_8_1_2", null ],
        [ "Release 1.8.1.1", "changelog.html#log_1_8_1_1", null ],
        [ "Release 1.8.1", "changelog.html#log_1_8_1", null ],
        [ "Release 1.8.0", "changelog.html#log_1_8_0", null ]
      ] ],
      [ "1.7 Series", "changelog.html#log_1_7", null ],
      [ "1.6 Series", "changelog.html#log_1_6", null ],
      [ "1.5 Series", "changelog.html#log_1_5", null ],
      [ "1.4 Series", "changelog.html#log_1_4", null ],
      [ "1.3 Series", "changelog.html#log_1_3", null ],
      [ "1.2 Series", "changelog.html#log_1_2", null ],
      [ "1.1 Series", "changelog.html#log_1_1", null ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"arch.html",
"commands.html#cmdthrows"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';