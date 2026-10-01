# SOURCE-PRUNE-DIAG-001 — admission failure, zero experimental calls

Registration8aab8171b774a12131b061dd15eda80378dde0f7 was pushed and exactremote verified. The single run invocation failed before its start marker and before any source/cipher experiment. The build check attempted `ROOT[s["path"]]`, subscripting a Path, instead of joining the artifact path.

The frozen implementation/protocol/build are preserved; no retry or complete audit is performed under001. [admission-failure.json](../../results/SOURCE-PRUNE-DIAG-001/admission-failure.json) records the error and zero calls. A separately registered002 adds the path fix and exercises actual input admission. Prior2656tests passed but mocked this admission boundary in tiny transport; that gap is explicitly retained. No diagnosis or recovery result was obtained.
