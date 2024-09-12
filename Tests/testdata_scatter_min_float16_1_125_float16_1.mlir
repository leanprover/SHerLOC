"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x125xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x125xf16>, tensor<1xf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x125xf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%7) : (tensor<f16>) -> ()
    }) : (tensor<1x125xf16>, tensor<1xi64>, tensor<1xf16>) -> tensor<1x125xf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1x125xf16>, tensor<1x125xf16>) -> ()
    "func.return"(%6) : (tensor<1x125xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x125xf16>, tensor<1xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xD9BE8FC2F93D2C44F03CFFC3D23C0640283CFDB50E4067400AB5A54443401ABC26C47E410FC2DCB1E8BC5EB8C73EE13ED537703EA542953F8E4331419840F2438E4478C7A83B28410A414D3EA1C006380142EDBAEEC5793E07457B45FC41E2344CC0CCBC36414840FD3D3FC6A64032C51DB9E0C51B37D0C01741AD3B26B6AB44B8B9774168C16FBC2444183777C214A4024271455ABA1C3FD9C5A343C3B4C9C778C3193F6A4445C5D43AE4C09437FF3D3243A2436E45E6C307BB27BBCBBFF7406F415F426340BBC203B5D13F1340E244393F19C13841373CE544C64266445CC13DBF28C4C7BF40C1384160C9A8451D466341D3C28A3924BC84BE"> : tensor<1x125xf16>}> : () -> tensor<1x125xf16>
    %2 = "stablehlo.constant"() <{value = dense<-2.517090e-01> : tensor<1xf16>}> : () -> tensor<1xf16>
    "func.return"(%1, %2) : (tensor<1x125xf16>, tensor<1xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x125xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xD9BE8FC2F93D2C44F03CFFC3D23C0640283CFDB50E4067400AB5A54443401ABC26C47E410FC2DCB1E8BC5EB8C73EE13ED537703EA542953F8E4331419840F2438E4478C7A83B28410A414D3EA1C006380142EDBAEEC5793E07457B45FC41E2344CC0CCBC36414840FD3D3FC6A64032C51DB9E0C51B37D0C01741AD3B26B6AB44B8B9774168C16FBC2444183777C214A4024271455ABA1C3FD9C5A343C3B4C9C778C3193F6A4445C5D43AE4C09437FF3D3243A2436E45E6C307BB27BBCBBFF7406F415F426340BBC203B5D13F1340E244393F19C13841373CE544C64266445CC13DBF28C4C7BF40C1384160C9A8451D466341D3C28A3924BC84BE"> : tensor<1x125xf16>}> : () -> tensor<1x125xf16>
    "func.return"(%0) : (tensor<1x125xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

