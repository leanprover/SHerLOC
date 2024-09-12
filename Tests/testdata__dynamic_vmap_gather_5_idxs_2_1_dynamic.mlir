"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x5xf32>, tensor<?x2x1xi64>) -> tensor<?x2xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x5xf32>, %arg2: tensor<?x2x1xi64>):
    %0 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %1 = "stablehlo.reshape"(%0) : (tensor<i32>) -> tensor<1xi32>
    %2 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.concatenate"(%1, %2, %3) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %5 = "stablehlo.dynamic_iota"(%4) <{iota_dimension = 0 : i64}> : (tensor<3xi32>) -> tensor<?x2x1xi64>
    %6 = "stablehlo.concatenate"(%5, %arg2) <{dimension = 2 : i64}> : (tensor<?x2x1xi64>, tensor<?x2x1xi64>) -> tensor<?x2x2xi64>
    %7 = "stablehlo.gather"(%arg1, %6) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1>}> : (tensor<?x5xf32>, tensor<?x2x2xi64>) -> tensor<?x2xf32>
    "func.return"(%7) : (tensor<?x2xf32>) -> ()
  }) : () -> ()
}) : () -> ()

