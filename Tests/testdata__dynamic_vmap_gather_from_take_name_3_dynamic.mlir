"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x10x10x10xf32>, tensor<?x2xi32>) -> tensor<?x2x10x10xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg3: tensor<i64>, %arg4: tensor<?x10x10x10xf32>, %arg5: tensor<?x2xi32>):
    %14 = "func.call"(%arg3, %arg4, %arg5) <{callee = @_take}> : (tensor<i64>, tensor<?x10x10x10xf32>, tensor<?x2xi32>) -> tensor<?x2x10x10xf32>
    "func.return"(%14) : (tensor<?x2x10x10xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?x10x10x10xf32>, tensor<?x2xi32>) -> tensor<?x2x10x10xf32>, sym_name = "_take", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x10x10x10xf32>, %arg2: tensor<?x2xi32>):
    %0 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %1 = "stablehlo.reshape"(%0) : (tensor<i32>) -> tensor<1xi32>
    %2 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.concatenate"(%1, %2, %3) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %5 = "stablehlo.dynamic_broadcast_in_dim"(%arg2, %4) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<?x2xi32>, tensor<3xi32>) -> tensor<?x2x1xi32>
    %6 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %7 = "stablehlo.reshape"(%6) : (tensor<i32>) -> tensor<1xi32>
    %8 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %9 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %10 = "stablehlo.concatenate"(%7, %8, %9) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %11 = "stablehlo.dynamic_iota"(%10) <{iota_dimension = 0 : i64}> : (tensor<3xi32>) -> tensor<?x2x1xi32>
    %12 = "stablehlo.concatenate"(%11, %5) <{dimension = 2 : i64}> : (tensor<?x2x1xi32>, tensor<?x2x1xi32>) -> tensor<?x2x2xi32>
    %13 = "stablehlo.gather"(%arg1, %12) <{dimension_numbers = #stablehlo.gather<offset_dims = [2, 3], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 2>, slice_sizes = array<i64: 1, 1, 10, 10>}> : (tensor<?x10x10x10xf32>, tensor<?x2x2xi32>) -> tensor<?x2x10x10xf32>
    "func.return"(%13) : (tensor<?x2x10x10xf32>) -> ()
  }) : () -> ()
}) : () -> ()

