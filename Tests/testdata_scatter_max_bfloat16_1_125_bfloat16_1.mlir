"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x125xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x125xbf16>, tensor<1xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x125xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%7) : (tensor<bf16>) -> ()
    }) : (tensor<1x125xbf16>, tensor<1xi64>, tensor<1xbf16>) -> tensor<1x125xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1x125xbf16>, tensor<1x125xbf16>) -> ()
    "func.return"(%6) : (tensor<1x125xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x125xbf16>, tensor<1xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x52BEAB403440B63E273F44BF2FC0FC3FC73FB13E6340C03FD6C0C9C0B240273F974093BFB03F3540A73E8B40123F66C08F3F95C0D6BF63C01EC0BFBF67BF0640EBBE073CEEBF01C0F740A2C0C1C03BC0FABC653FF53F133F873F1B401B4079BFC4BFD34017C0AC3F8FC090C05AC0C8BEB8BE10C0D33F0CC0A0BFE23E3A4021BF6940813D3240F04089C002BFB03F62C0F44041C00A400E402B3F21401240B43E573EDABF3E401CBF1D40C640523FC4BF48BF71C00E40553F973FDEBF06407DC0354004C0C8BFD1C021C0F83F4D404A400D401A4025C002C0AFBF2F3E1DBFF6401FBF363F0BC02EBE14BF2DBF1CBF0B402EC0BDBF0B4066400E3F"> : tensor<1x125xbf16>}> : () -> tensor<1x125xbf16>
    %2 = "stablehlo.constant"() <{value = dense<1.531250e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
    "func.return"(%1, %2) : (tensor<1x125xbf16>, tensor<1xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x125xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xC43FAB403440B63E273F44BF2FC0FC3FC73FB13E6340C03FD6C0C9C0B240273F974093BFB03F3540A73E8B40123F66C08F3F95C0D6BF63C01EC0BFBF67BF0640EBBE073CEEBF01C0F740A2C0C1C03BC0FABC653FF53F133F873F1B401B4079BFC4BFD34017C0AC3F8FC090C05AC0C8BEB8BE10C0D33F0CC0A0BFE23E3A4021BF6940813D3240F04089C002BFB03F62C0F44041C00A400E402B3F21401240B43E573EDABF3E401CBF1D40C640523FC4BF48BF71C00E40553F973FDEBF06407DC0354004C0C8BFD1C021C0F83F4D404A400D401A4025C002C0AFBF2F3E1DBFF6401FBF363F0BC02EBE14BF2DBF1CBF0B402EC0BDBF0B4066400E3F"> : tensor<1x125xbf16>}> : () -> tensor<1x125xbf16>
    "func.return"(%0) : (tensor<1x125xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

