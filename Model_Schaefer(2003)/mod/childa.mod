COMMENT
iterator for traversing all the daughters of the currently accessed section

section subtree_traverse("statement")

executes statement for every daughter of section.
Just before the statement is executed the currently accessed section is set.

ENDCOMMENT

NEURON {
        SUFFIX nothing
}

VERBATIM
#ifdef NRN_MECHANISM_DATA_IS_SOA
#define get_child(sec) _nrn_mechanism_get_child(sec)
#define get_sibling(sec) _nrn_mechanism_get_sibling(sec)
#else
#define get_child(sec) sec->child
#define get_sibling(sec) sec->sibling
#endif
static void subtree(Section* sec, Symbol* sym) {
        for (Section* child = get_child(sec); child; child = get_sibling(child)) {
       nrn_pushsec(child);       /* move these three (sec becomes child) */
        hoc_run_stmt(sym);      /* into the loop to do only the first level */
        nrn_popsec(); 

        }
}
#ifndef NRN_VERSION_GTEQ_8_2_0
Section* chk_access();
Symbol* hoc_parse_stmt();
#endif
ENDVERBATIM

PROCEDURE subtree_traverse() {
  VERBATIM
  {
        Symlist* symlist = (Symlist*)0;
        subtree(chk_access(), hoc_parse_stmt(gargstr(1), &symlist));
        /* if following not executed (ie hoc error in statement),
           some memory will leak */
        hoc_free_list(&symlist);
  }
  ENDVERBATIM
}
