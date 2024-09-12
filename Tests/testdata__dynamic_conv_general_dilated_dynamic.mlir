"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x3x9x10xf32>, tensor<3x3x4x5xf32>) -> tensor<?x3x3x1xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x3x9x10xf32>, %arg2: tensor<3x3x4x5xf32>):
    %0 = "stablehlo.convolution"(%arg1, %arg2) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>, feature_group_count = 1 : i64, rhs_dilation = array<i64: 1, 2>, window_strides = array<i64: 2, 3>}> : (tensor<?x3x9x10xf32>, tensor<3x3x4x5xf32>) -> tensor<?x3x3x1xf32>
    "func.return"(%0) : (tensor<?x3x3x1xf32>) -> ()
  }) : () -> ()
}) : () -> ()

