"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x100xi32>, tensor<?x100xi32>) -> (tensor<?x100xi32>, tensor<?x100xi32>), sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x100xi32>, %arg2: tensor<?x100xi32>):
    %0:2 = "stablehlo.sort"(%arg1, %arg2) <{dimension = 1 : i64, is_stable = true}> ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>, %arg6: tensor<i32>):
      %1 = "stablehlo.compare"(%arg3, %arg4) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "stablehlo.return"(%1) : (tensor<i1>) -> ()
    }) : (tensor<?x100xi32>, tensor<?x100xi32>) -> (tensor<?x100xi32>, tensor<?x100xi32>)
    "func.return"(%0#0, %0#1) : (tensor<?x100xi32>, tensor<?x100xi32>) -> ()
  }) : () -> ()
}) : () -> ()

