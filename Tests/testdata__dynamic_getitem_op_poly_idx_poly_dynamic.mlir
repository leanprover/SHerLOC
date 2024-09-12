"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x4xf32>, tensor<?xi32>) -> tensor<?x4xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x4xf32>, %arg2: tensor<?xi32>):
    %0 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %2 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %3 = "stablehlo.reshape"(%2) : (tensor<i32>) -> tensor<1xi32>
    %4 = "stablehlo.dynamic_broadcast_in_dim"(%1, %3) <{broadcast_dimensions = array<i64>}> : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
    %5 = "stablehlo.compare"(%arg2, %4) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi1>
    %6 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %7 = "stablehlo.reshape"(%6) : (tensor<i32>) -> tensor<1xi32>
    %8 = "stablehlo.dynamic_broadcast_in_dim"(%0, %7) <{broadcast_dimensions = array<i64>}> : (tensor<i32>, tensor<1xi32>) -> tensor<?xi32>
    %9 = "stablehlo.add"(%arg2, %8) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    %10 = "stablehlo.select"(%5, %9, %arg2) : (tensor<?xi1>, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    %11 = "stablehlo.convert"(%10) : (tensor<?xi32>) -> tensor<?xi64>
    %12 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %13 = "stablehlo.reshape"(%12) : (tensor<i32>) -> tensor<1xi32>
    %14 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %15 = "stablehlo.concatenate"(%13, %14) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %16 = "stablehlo.dynamic_broadcast_in_dim"(%11, %15) <{broadcast_dimensions = array<i64: 0>}> : (tensor<?xi64>, tensor<2xi32>) -> tensor<?x1xi64>
    %17 = "stablehlo.gather"(%arg1, %16) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 4>}> : (tensor<?x4xf32>, tensor<?x1xi64>) -> tensor<?x4xf32>
    "func.return"(%17) : (tensor<?x4xf32>) -> ()
  }) : () -> ()
}) : () -> ()

