"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x3xf32>, tensor<?x1xi64>) -> tensor<?x1xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x3xf32>, %arg2: tensor<?x1xi64>):
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %2 = "stablehlo.concatenate"(%0, %1) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %3 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %4 = "stablehlo.reshape"(%3) : (tensor<i32>) -> tensor<1xi32>
    %5 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %6 = "stablehlo.concatenate"(%4, %5) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %7 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %8 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %9 = "stablehlo.concatenate"(%7, %8) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %10 = "stablehlo.real_dynamic_slice"(%arg2, %2, %6, %9) : (tensor<?x1xi64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x1xi64>
    %11 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %12 = "stablehlo.reshape"(%11) : (tensor<i32>) -> tensor<1xi32>
    %13 = "stablehlo.dynamic_reshape"(%10, %12) : (tensor<?x1xi64>, tensor<1xi32>) -> tensor<?xi64>
    %14 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %15 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %16 = "stablehlo.reshape"(%15) : (tensor<i32>) -> tensor<1xi32>
    %17 = "stablehlo.dynamic_broadcast_in_dim"(%14, %16) <{broadcast_dimensions = array<i64>}> : (tensor<i64>, tensor<1xi32>) -> tensor<?xi64>
    %18 = "stablehlo.compare"(%13, %17) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi1>
    %19 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
    %20 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %21 = "stablehlo.reshape"(%20) : (tensor<i32>) -> tensor<1xi32>
    %22 = "stablehlo.dynamic_broadcast_in_dim"(%19, %21) <{broadcast_dimensions = array<i64>}> : (tensor<i64>, tensor<1xi32>) -> tensor<?xi64>
    %23 = "stablehlo.add"(%13, %22) : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
    %24 = "stablehlo.select"(%18, %23, %13) : (tensor<?xi1>, tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
    %25 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %26 = "stablehlo.reshape"(%25) : (tensor<i32>) -> tensor<1xi32>
    %27 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %28 = "stablehlo.concatenate"(%26, %27) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %29 = "stablehlo.dynamic_broadcast_in_dim"(%24, %28) <{broadcast_dimensions = array<i64: 0>}> : (tensor<?xi64>, tensor<2xi32>) -> tensor<?x1xi64>
    %30 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %31 = "stablehlo.reshape"(%30) : (tensor<i32>) -> tensor<1xi32>
    %32 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %33 = "stablehlo.concatenate"(%31, %32) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %34 = "stablehlo.dynamic_iota"(%33) <{iota_dimension = 0 : i64}> : (tensor<2xi32>) -> tensor<?x1xi64>
    %35 = "stablehlo.concatenate"(%34, %29) <{dimension = 1 : i64}> : (tensor<?x1xi64>, tensor<?x1xi64>) -> tensor<?x2xi64>
    %36 = "stablehlo.gather"(%arg1, %35) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = true, slice_sizes = array<i64: 1, 1>}> : (tensor<?x3xf32>, tensor<?x2xi64>) -> tensor<?x1xf32>
    "func.return"(%36) : (tensor<?x1xf32>) -> ()
  }) : () -> ()
}) : () -> ()

