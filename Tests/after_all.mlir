"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "after_all_op_test"}> ({
    %0 = "stablehlo.after_all"() : () -> !stablehlo.token
    %1 = "stablehlo.after_all"() : () -> !stablehlo.token
    %2 = "stablehlo.after_all"(%0, %1) : (!stablehlo.token, !stablehlo.token) -> !stablehlo.token
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

