"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x125xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x125xbf16>, tensor<1xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x125xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%7) : (tensor<bf16>) -> ()
    }) : (tensor<1x125xbf16>, tensor<1xi64>, tensor<1xbf16>) -> tensor<1x125xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1x125xbf16>, tensor<1x125xbf16>) -> ()
    "func.return"(%6) : (tensor<1x125xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x125xbf16>, tensor<1xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x1DC004C040C0BB3FB6409BBF1DC02A40EABD2CC0E1BF16C0ACBE10BF2D409C40C2BFEF3F86C01CC04E4098C02940903EBBBE7A3F32C042BF83C0E2BF9E3E1C400740113F283F413FA1BFACC03A3F6EC088C001410EBF4BBF99C0A1C08A3FBD40BB3F5D4086C0914039BF4E402EC0133F0BC07BBF82BD40BE86402EBF43C00840CB3F74C0A0C0ABBEA03FB43E9EC00BC056405EC0D4BFEBBF813E2C40B93E0EC001C08FC0C2BD8D40643FE33F78BF723F2C40E6BFAD3F2CBFC6BFCDC052BBEC3F90BEA9BFF9C020C06EC00C411EC0C1BF2C40EDBF15408E3FF6BF234031BF1E4047C005BF593F213F614052BFF3BF05BF773FE54027C046C06D40"> : tensor<1x125xbf16>}> : () -> tensor<1x125xbf16>
    %2 = "stablehlo.constant"() <{value = dense<-1.429690e+00> : tensor<1xbf16>}> : () -> tensor<1xbf16>
    "func.return"(%1, %2) : (tensor<1x125xbf16>, tensor<1xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x125xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x1DC004C040C0BB3FB6409BBF1DC02A40EABD2CC0E1BF16C0ACBE10BF2D409C40C2BFEF3F86C01CC04E4098C02940903EBBBE7A3F32C042BF83C0E2BF9E3E1C400740113F283F413FA1BFACC03A3F6EC088C001410EBF4BBF99C0A1C08A3FBD40BB3F5D4086C0914039BF4E402EC0133F0BC07BBF82BD40BE86402EBF43C00840CB3F74C0A0C0ABBEA03FB43E9EC00BC056405EC0D4BFEBBF813E2C40B93E0EC001C08FC0C2BD8D40643FE33F78BF723F2C40E6BFAD3F2CBFC6BFCDC052BBEC3F90BEA9BFF9C020C06EC00C411EC0C1BF2C40EDBF15408E3FF6BF234031BF1E4047C005BF593F213F614052BFF3BF05BF773FE54027C046C06D40"> : tensor<1x125xbf16>}> : () -> tensor<1x125xbf16>
    "func.return"(%0) : (tensor<1x125xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

