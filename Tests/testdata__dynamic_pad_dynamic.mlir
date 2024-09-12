"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x2x5xf32>, tensor<f32>) -> tensor<?x2x11xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x2x5xf32>, %arg2: tensor<f32>):
    %0 = "stablehlo.pad"(%arg1, %arg2) <{edge_padding_high = array<i64: 0, 0, 1>, edge_padding_low = array<i64: 0, 0, 1>, interior_padding = array<i64: 0, 0, 1>}> : (tensor<?x2x5xf32>, tensor<f32>) -> tensor<?x2x11xf32>
    "func.return"(%0) : (tensor<?x2x11xf32>) -> ()
  }) : () -> ()
}) : () -> ()

