"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x20x20xi32>, %arg2: tensor<?x20x20xi32>):
    %0 = "stablehlo.and"(%arg1, %arg2) : (tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi32>
    "func.return"(%0) : (tensor<?x20x20xi32>) -> ()
  }) : () -> ()
}) : () -> ()

