"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x1x16x2xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x1x16x1xf32>, tensor<4x1x1x2xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<1x1x16x2xf32>
    %5 = "stablehlo.convolution"(%3#0, %3#1) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, padding = dense<[[1, 2], [0, 0]]> : tensor<2x2xi64>}> : (tensor<1x1x16x1xf32>, tensor<4x1x1x2xf32>) -> tensor<1x1x16x2xf32>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1x1x16x2xf32>, tensor<1x1x16x2xf32>) -> ()
    "func.return"(%5) : (tensor<1x1x16x2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x1x16x1xf32>, tensor<4x1x1x2xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[[0.249547243], [-3.84318519], [-1.0406214], [1.16896808], [-0.0966261476], [0.145545349], [-0.652937531], [4.2463026], [-1.61683857], [-2.21878624], [-6.34280634], [-1.90672576], [-4.64762354], [3.25072265], [-0.142132208], [-0.168175161]]]]> : tensor<1x1x16x1xf32>}> : () -> tensor<1x1x16x1xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[[[0.354699761, 0.679385543]]], [[[0.609961211, 0.152189493]]], [[[-2.194420e+00, -2.72838569]]], [[[-0.477422208, -1.72463417]]]]> : tensor<4x1x1x2xf32>}> : () -> tensor<4x1x1x2xf32>
    "func.return"(%1, %2) : (tensor<1x1x16x1xf32>, tensor<4x1x1x2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x1x16x2xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[[0.15221414, 0.0379784666], [-2.34419394, -0.584892392], [-0.634738684, -0.158371642], [0.713025212, 0.177904665], [-5.893820e-02, -0.0147054847], [0.0887770206, 0.0221504737], [-0.398266554, -0.0993702337], [2.59007978, 0.646242618], [-0.986208796, -0.24606584], [-1.35337353, -0.337675959], [-3.86886573, -0.965308487], [-1.16302872, -0.290183634], [-2.834870e+00, -0.707319498], [1.98281467, 0.494725823], [-0.0866951346, -0.0216310285], [-0.102580324, -0.0255944934]]]]> : tensor<1x1x16x2xf32>}> : () -> tensor<1x1x16x2xf32>
    "func.return"(%0) : (tensor<1x1x16x2xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

