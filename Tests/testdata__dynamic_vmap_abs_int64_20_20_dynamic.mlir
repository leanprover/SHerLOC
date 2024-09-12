"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x20x20xi64>) -> tensor<?x20x20xi64>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x20x20xi64>):
    %0 = "stablehlo.abs"(%arg1) : (tensor<?x20x20xi64>) -> tensor<?x20x20xi64>
    "func.return"(%0) : (tensor<?x20x20xi64>) -> ()
  }) : () -> ()
}) : () -> ()

