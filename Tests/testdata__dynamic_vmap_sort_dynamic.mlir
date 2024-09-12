"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x5x7xi32>) -> tensor<?x5x7xi32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x5x7xi32>):
    %0 = "stablehlo.sort"(%arg1) <{dimension = 1 : i64}> ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = "stablehlo.compare"(%arg2, %arg3) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "stablehlo.return"(%1) : (tensor<i1>) -> ()
    }) : (tensor<?x5x7xi32>) -> tensor<?x5x7xi32>
    "func.return"(%0) : (tensor<?x5x7xi32>) -> ()
  }) : () -> ()
}) : () -> ()

