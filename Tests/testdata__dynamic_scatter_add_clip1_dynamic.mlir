"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x4xf32>, tensor<?x2xi32>, tensor<?x1xf32>) -> tensor<?x4xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x4xf32>, %arg2: tensor<?x2xi32>, %arg3: tensor<?x1xf32>):
    %0 = "stablehlo.constant"() <{value = dense<-1> : tensor<i64>}> : () -> tensor<i64>
    %1 = "stablehlo.add"(%arg0, %0) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %2 = "stablehlo.convert"(%1) : (tensor<i64>) -> tensor<i64>
    %3 = "stablehlo.broadcast_in_dim"(%2) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %4 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
    %5 = "stablehlo.broadcast_in_dim"(%4) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<1xi64>
    %6 = "stablehlo.concatenate"(%3, %5) <{dimension = 0 : i64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
    %7 = "stablehlo.constant"() <{value = dense<2147483647> : tensor<ui64>}> : () -> tensor<ui64>
    %8 = "stablehlo.convert"(%7) : (tensor<ui64>) -> tensor<i64>
    %9 = "stablehlo.broadcast_in_dim"(%8) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<2xi64>
    %10 = "stablehlo.minimum"(%6, %9) : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
    %11 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %12 = "stablehlo.reshape"(%11) : (tensor<i32>) -> tensor<1xi32>
    %13 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %14 = "stablehlo.concatenate"(%12, %13) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %15 = "stablehlo.dynamic_broadcast_in_dim"(%10, %14) <{broadcast_dimensions = array<i64: 1>}> : (tensor<2xi64>, tensor<2xi32>) -> tensor<?x2xi64>
    %16 = "stablehlo.convert"(%arg2) : (tensor<?x2xi32>) -> tensor<?x2xi64>
    %17 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %18 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %19 = "stablehlo.reshape"(%18) : (tensor<i32>) -> tensor<1xi32>
    %20 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %21 = "stablehlo.concatenate"(%19, %20) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %22 = "stablehlo.dynamic_broadcast_in_dim"(%17, %21) <{broadcast_dimensions = array<i64>}> : (tensor<i64>, tensor<2xi32>) -> tensor<?x2xi64>
    %23 = "stablehlo.clamp"(%22, %16, %15) : (tensor<?x2xi64>, tensor<?x2xi64>, tensor<?x2xi64>) -> tensor<?x2xi64>
    %24 = "stablehlo.scatter"(%arg1, %23, %arg3) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
      %25 = "stablehlo.add"(%arg4, %arg5) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%25) : (tensor<f32>) -> ()
    }) : (tensor<?x4xf32>, tensor<?x2xi64>, tensor<?x1xf32>) -> tensor<?x4xf32>
    "func.return"(%24) : (tensor<?x4xf32>) -> ()
  }) : () -> ()
}) : () -> ()

