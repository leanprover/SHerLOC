"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x50x3xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<32> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x50x3xbf16>, tensor<1x3xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x50x3xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%7) : (tensor<bf16>) -> ()
    }) : (tensor<1x50x3xbf16>, tensor<1xi64>, tensor<1x3xbf16>) -> tensor<1x50x3xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1x50x3xbf16>, tensor<1x50x3xbf16>) -> ()
    "func.return"(%6) : (tensor<1x50x3xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x50x3xbf16>, tensor<1x3xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFDBFC03F14C06DC003C03F3E633F95C0CE3F40C0E1BF31BF0CC0A83F7EBF813D81401B3F553F0640FA3F04403C4096BF06BF9740833F44C0AFBF003FFABE09C0E63FACC08DBE44BF65BF843F0840F13F9AC0A1C05540B13F4B40BCBF99BFDCBFAFBF1B407F40AD409840B3C0B13E654054C089BF92BD6CC04140F23FD3BFAFBFF9BE714015BFCD3F36401A406ABFD7BF594027C046BFC7BF483E4E3F83405040A2401A402ABF1DC02040C7BFB2BF353FA4BF1540743E723FB3BF51408A3EBF407E3E9340EBBD5DC077BFD03F0AC0D0408F3F84BE1BBF3FC0533FA5BF3D3F4640243E3A3F57C058BF47C0A0BF91C00CBFFF3FBEBF4EBEC2BC944085BF92BF09C035BF9A3EA7BD3D3ED9BE19408A403AC00FC0A83F62408AC03DC0F0BFA5C044C07CBFC6BE1340BFBD47C02240"> : tensor<1x50x3xbf16>}> : () -> tensor<1x50x3xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[1.367190e+00, 1.171880e+00, -1.484380e+00]]> : tensor<1x3xbf16>}> : () -> tensor<1x3xbf16>
    "func.return"(%1, %2) : (tensor<1x50x3xbf16>, tensor<1x3xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x50x3xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFDBFC03F14C06DC003C03F3E633F95C0CE3F40C0E1BF31BF0CC0A83F7EBF813D81401B3F553F0640FA3F04403C4096BF06BF9740833F44C0AFBF003FFABE09C0E63FACC08DBE44BF65BF843F0840F13F9AC0A1C05540B13F4B40BCBF99BFDCBFAFBF1B407F40AD409840B3C0B13E654054C089BF92BD6CC04140F23FD3BFAFBFF9BE714015BFCD3F36401A406ABFD7BF594027C046BFC7BF483E4E3F83405040A2401A402ABF1DC02040C7BFB2BF353FA4BF1540743E723FB3BF51408A3EBF40AF3F9340EBBD5DC077BFD03F0AC0D0408F3F84BE1BBF3FC0533FA5BF3D3F4640243E3A3F57C058BF47C0A0BF91C00CBFFF3FBEBF4EBEC2BC944085BF92BF09C035BF9A3EA7BD3D3ED9BE19408A403AC00FC0A83F62408AC03DC0F0BFA5C044C07CBFC6BE1340BFBD47C02240"> : tensor<1x50x3xbf16>}> : () -> tensor<1x50x3xbf16>
    "func.return"(%0) : (tensor<1x50x3xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

