"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1xi32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<2x1xi64>}> : () -> tensor<2x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1xi32>, tensor<2xi32>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1xi32>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "stablehlo.return"(%7) : (tensor<i32>) -> ()
    }) : (tensor<1xi32>, tensor<2x1xi64>, tensor<2xi32>) -> tensor<1xi32>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1xi32>, tensor<1xi32>) -> ()
    "func.return"(%6) : (tensor<1xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1xi32>, tensor<2xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %2 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xi32>}> : () -> tensor<2xi32>
    "func.return"(%1, %2) : (tensor<1xi32>, tensor<2xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1xi32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    "func.return"(%0) : (tensor<1xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

