"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x50x3xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<32> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x50x3xf16>, tensor<1x3xf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x50x3xf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%7) : (tensor<f16>) -> ()
    }) : (tensor<1x50x3xf16>, tensor<1xi64>, tensor<1x3xf16>) -> tensor<1x50x3xf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1x50x3xf16>, tensor<1x50x3xf16>) -> ()
    "func.return"(%6) : (tensor<1x50x3xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x50x3xf16>, tensor<1x3xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x28BF47C11BC4D4C344BD4DC1B43AF5442331D1B9DBC2A540EBC5FCB622BE53377E3406C2E844FF416FC2ACC38EC0A5C77EBE0F450EB62FC3813F133A0CC02CC0AC37F7C438436AAC1940F5BD2E452AC09641D6C0CCC07CAF79439BC1C2403040C83CB240A4C4BABB6B4157C0D5C474BD4E43734139BCECBCC0C57241AE3FB53D493C2444044014390EC354BB31C325C67FB34AC2AEC4EFC8A8C3364332BC95C4CDC40BB9B0BFAFC63CBEB34458BC914471B08DBBBDC2B8C39DBD503AFE42F2C089BE37C018C1AB40EC3D023DE441584229BFBCC068B9BCC671C106C66ABC06B4663195449CC1ADB98F351940FC3F68B7A64771C22634023D6744E4312F39673CB8B4603F27C20CBC41C2B54274C2C4C377BEFAC59542EA3C1B44673C4FBA1E41573FF642DC3FDC4826C0EFC3"> : tensor<1x50x3xf16>}> : () -> tensor<1x50x3xf16>
    %2 = "stablehlo.constant"() <{value = dense<[[2.994140e+00, -5.250000e+00, 5.581050e-01]]> : tensor<1x3xf16>}> : () -> tensor<1x3xf16>
    "func.return"(%1, %2) : (tensor<1x50x3xf16>, tensor<1x3xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x50x3xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x28BF47C11BC4D4C344BD4DC1B43AF5442331D1B9DBC2A540EBC5FCB622BE53377E3406C2E844FF416FC2ACC38EC0A5C77EBE0F450EB62FC3813F133A0CC02CC0AC37F7C438436AAC1940F5BD2E452AC09641D6C0CCC07CAF79439BC1C2403040C83CB240A4C4BABB6B4157C0D5C474BD4E43734139BCECBCC0C57241AE3FB53D493C2444044014390EC354BB31C325C67FB34AC2AEC4EFC8A8C3364332BC95C4CDC40BB9B0BFAFC63CBEB34458BC914471B08DBBBDC2B8C39DBD503AFE42F2C0FD4137C07738AB40EC3D023DE441584229BFBCC068B9BCC671C106C66ABC06B4663195449CC1ADB98F351940FC3F68B7A64771C22634023D6744E4312F39673CB8B4603F27C20CBC41C2B54274C2C4C377BEFAC59542EA3C1B44673C4FBA1E41573FF642DC3FDC4826C0EFC3"> : tensor<1x50x3xf16>}> : () -> tensor<1x50x3xf16>
    "func.return"(%0) : (tensor<1x50x3xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

