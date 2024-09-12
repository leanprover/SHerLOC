"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<1x?x16xf32>, tensor<4x16x16xf32>) -> tensor<1x?x16xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<1x?x16xf32>, %arg2: tensor<4x16x16xf32>):
    %0 = "stablehlo.convolution"(%arg1, %arg2) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>, feature_group_count = 1 : i64, padding = dense<[[1, 2]]> : tensor<1x2xi64>}> : (tensor<1x?x16xf32>, tensor<4x16x16xf32>) -> tensor<1x?x16xf32>
    "func.return"(%0) : (tensor<1x?x16xf32>) -> ()
  }) : () -> ()
}) : () -> ()

