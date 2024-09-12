"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[3, 2]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3xf32>, tensor<2xf32>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3xf32>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%7) : (tensor<f32>) -> ()
    }) : (tensor<4x2x3xf32>, tensor<2xi64>, tensor<2xf32>) -> tensor<4x2x3xf32>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x2x3xf32>, tensor<4x2x3xf32>) -> ()
    "func.return"(%6) : (tensor<4x2x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3xf32>, tensor<2xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[0.93096745, -2.901610e+00, -0.136921942], [-1.80305481, 0.274286896, -2.30252504]], [[-4.93505478, 1.80857766, 1.2502718], [7.76369047, -5.67987299, -2.081790e+00]], [[-2.27660036, 1.0958333, 0.296646416], [3.69325113, -4.21838379, 2.43013287]], [[-3.66318417, 0.186973795, -1.25945151], [-1.06467712, 3.4913063, 1.16593659]]]> : tensor<4x2x3xf32>}> : () -> tensor<4x2x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<[-2.92589569, 2.0510335]> : tensor<2xf32>}> : () -> tensor<2xf32>
    "func.return"(%1, %2) : (tensor<4x2x3xf32>, tensor<2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[0.93096745, -2.901610e+00, -0.136921942], [-1.80305481, 0.274286896, -2.30252504]], [[-4.93505478, 1.80857766, 1.2502718], [7.76369047, -5.67987299, -2.081790e+00]], [[-2.27660036, 1.0958333, 0.296646416], [3.69325113, -4.21838379, 2.43013287]], [[-3.66318417, 0.186973795, -2.92589569], [-1.06467712, 3.4913063, 1.16593659]]]> : tensor<4x2x3xf32>}> : () -> tensor<4x2x3xf32>
    "func.return"(%0) : (tensor<4x2x3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

