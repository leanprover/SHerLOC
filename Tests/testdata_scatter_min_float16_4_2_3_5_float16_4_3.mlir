"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xf16>, tensor<4x3xf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%7) : (tensor<f16>) -> ()
    }) : (tensor<4x2x3x5xf16>, tensor<2xi64>, tensor<4x3xf16>) -> tensor<4x2x3x5xf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x2x3x5xf16>, tensor<4x2x3x5xf16>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xf16>, tensor<4x3xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xD7C08943F2394D3066BD763C05B8B2C5ED3E923869C2A4428CBC0E44E73DC9C0D24534C301C871BB213C0D436BC207C58D373B419F44B7B98247C544DAC493C264C0A0BD6636743EA9C09B4337BF1B3D3D4117BF53C7C539D8BBEAB9B4C4D1C5554446C08D45D5BF4644EC377C38C4C287C054BA3EC000C2A94230C00148D1C2333D9FB5C246D54529C311BC8B441BC0E7C0ACBFB5B67EC4DEC105C374C0DF4359410F453540BABADA3EDBB57CC413C0954004402D3C0DC21BB13541E5281E3FC13301399C382DC449C4FBC0963D5F40B8404BBFD1C44AC583B6AA43E3C324BD5AC211C235C04346AF3D03C4FB4276BC"> : tensor<4x2x3x5xf16>}> : () -> tensor<4x2x3x5xf16>
    %2 = "stablehlo.constant"() <{value = dense<[[-2.015630e+00, 2.941410e+00, 5.554690e+00], [3.673830e+00, -5.297850e-01, 4.468750e+00], [6.342770e-01, -6.309500e-03, -8.105460e-01], [-4.644530e+00, -4.619140e-01, -1.895510e+00]]> : tensor<4x3xf16>}> : () -> tensor<4x3xf16>
    "func.return"(%1, %2) : (tensor<4x2x3x5xf16>, tensor<4x3xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xD7C08943F2394D3008C0763C05B8B2C5ED3E923869C2A4428CBC0E44E73DC9C0D24534C301C871BB213C0D436BC207C58D373B419F44B7B98247C544DAC493C264C0A0BD6636743EA9C09B4337BF3DB83D4117BF53C7C539D8BBEAB9B4C4D1C5554446C08D45D5BF4644EC377C38C4C287C054BA3EC000C2A94230C00148D1C213399FB5C246D54529C311BC8B441BC0E7C0ACBF7CBA7EC4DEC105C374C0DF4359410F453540BABADA3EDBB57CC413C0954004402D3C0DC21BB13541A5C41E3FC13301399C382DC449C4FBC0963D5F4095BF4BBFD1C44AC583B6AA43E3C324BD5AC211C235C04346AF3D03C4FB4276BC"> : tensor<4x2x3x5xf16>}> : () -> tensor<4x2x3x5xf16>
    "func.return"(%0) : (tensor<4x2x3x5xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

