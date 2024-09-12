"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x4x6xi32>, tensor<?x4x6xi32>) -> (tensor<?xi32>, tensor<?xi32>), sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x4x6xi32>, %arg2: tensor<?x4x6xi32>):
    %0 = "stablehlo.constant"() <{value = dense<3> : tensor<i32>}> : () -> tensor<i32>
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %2:2 = "stablehlo.reduce"(%arg1, %arg2, %0, %1) <{dimensions = array<i64: 1, 2>}> ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>, %arg6: tensor<i32>):
      %3 = "stablehlo.maximum"(%arg3, %arg5) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %4 = "stablehlo.minimum"(%arg4, %arg6) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "stablehlo.return"(%3, %4) : (tensor<i32>, tensor<i32>) -> ()
    }) : (tensor<?x4x6xi32>, tensor<?x4x6xi32>, tensor<i32>, tensor<i32>) -> (tensor<?xi32>, tensor<?xi32>)
    "func.return"(%2#0, %2#1) : (tensor<?xi32>, tensor<?xi32>) -> ()
  }) : () -> ()
}) : () -> ()

