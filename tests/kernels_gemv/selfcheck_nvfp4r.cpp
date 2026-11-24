#include <cstdio>
#include "ink/nvfp4_repack.h"
int main() { int rc = ink::nvfp4r_selfcheck(); printf("nvfp4r selfcheck: %s (%d)\n", rc ? "FAIL" : "OK", rc); return rc != 0; }
