"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x10x10x10xf32>, tensor<?x2x2xi32>) -> tensor<?x2x2x10x10xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg3: tensor<i64>, %arg4: tensor<?x10x10x10xf32>, %arg5: tensor<?x2x2xi32>):
    %16 = "func.call"(%arg3, %arg4, %arg5) <{callee = @_take}> : (tensor<i64>, tensor<?x10x10x10xf32>, tensor<?x2x2xi32>) -> tensor<?x2x2x10x10xf32>
    "func.return"(%16) : (tensor<?x2x2x10x10xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?x10x10x10xf32>, tensor<?x2x2xi32>) -> tensor<?x2x2x10x10xf32>, sym_name = "_take", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x10x10x10xf32>, %arg2: tensor<?x2x2xi32>):
    %0 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %1 = "stablehlo.reshape"(%0) : (tensor<i32>) -> tensor<1xi32>
    %2 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = "stablehlo.concatenate"(%1, %2, %3, %4) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %6 = "stablehlo.dynamic_broadcast_in_dim"(%arg2, %5) <{broadcast_dimensions = array<i64: 0, 1, 2>}> : (tensor<?x2x2xi32>, tensor<4xi32>) -> tensor<?x2x2x1xi32>
    %7 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %8 = "stablehlo.reshape"(%7) : (tensor<i32>) -> tensor<1xi32>
    %9 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %10 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %11 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %12 = "stablehlo.concatenate"(%8, %9, %10, %11) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %13 = "stablehlo.dynamic_iota"(%12) <{iota_dimension = 0 : i64}> : (tensor<4xi32>) -> tensor<?x2x2x1xi32>
    %14 = "stablehlo.concatenate"(%13, %6) <{dimension = 3 : i64}> : (tensor<?x2x2x1xi32>, tensor<?x2x2x1xi32>) -> tensor<?x2x2x2xi32>
    %15 = "stablehlo.gather"(%arg1, %14) <{dimension_numbers = #stablehlo.gather<offset_dims = [3, 4], collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 3>, slice_sizes = array<i64: 1, 1, 10, 10>}> : (tensor<?x10x10x10xf32>, tensor<?x2x2x2xi32>) -> tensor<?x2x2x10x10xf32>
    "func.return"(%15) : (tensor<?x2x2x10x10xf32>) -> ()
  }) : () -> ()
}) : () -> ()

