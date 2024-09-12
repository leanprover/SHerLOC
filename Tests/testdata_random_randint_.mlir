"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<32xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %270 = "func.call"() <{callee = @expected}> : () -> tensor<32xi8>
    %271 = "stablehlo.constant"() <{value = dense<42> : tensor<i64>}> : () -> tensor<i64>
    %272 = "stablehlo.constant"() <{value = dense<32> : tensor<i64>}> : () -> tensor<i64>
    %273 = "stablehlo.shift_right_logical"(%271, %272) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %274 = "stablehlo.convert"(%273) : (tensor<i64>) -> tensor<ui32>
    %275 = "stablehlo.broadcast_in_dim"(%274) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %276 = "stablehlo.constant"() <{value = dense<4294967295> : tensor<ui32>}> : () -> tensor<ui32>
    %277 = "stablehlo.convert"(%276) : (tensor<ui32>) -> tensor<i64>
    %278 = "stablehlo.and"(%271, %277) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %279 = "stablehlo.convert"(%278) : (tensor<i64>) -> tensor<ui32>
    %280 = "stablehlo.broadcast_in_dim"(%279) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<1xui32>
    %281 = "stablehlo.concatenate"(%275, %280) <{dimension = 0 : i64}> : (tensor<1xui32>, tensor<1xui32>) -> tensor<2xui32>
    %282 = "stablehlo.constant"() <{value = dense<-5> : tensor<i64>}> : () -> tensor<i64>
    %283 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
    %284 = "func.call"(%281, %282, %283) <{callee = @_randint}> : (tensor<2xui32>, tensor<i64>, tensor<i64>) -> tensor<32xi8>
    "stablehlo.custom_call"(%284, %270) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<32xi8>, tensor<32xi8>) -> ()
    "func.return"(%284) : (tensor<32xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<32xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %269 = "stablehlo.constant"() <{value = dense<[2, 2, 2, -1, 1, 0, 2, -3, -1, 1, 3, -2, -4, 1, -1, -1, 4, -5, 3, 3, 2, 3, 3, -5, 2, 4, 2, -5, 4, -1, -4, -2]> : tensor<32xi8>}> : () -> tensor<32xi8>
    "func.return"(%269) : (tensor<32xi8>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], function_type = (tensor<2xui32>, tensor<i64>, tensor<i64>) -> tensor<32xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_randint", sym_visibility = "private"}> ({
  ^bb0(%arg41: tensor<2xui32>, %arg42: tensor<i64>, %arg43: tensor<i64>):
    %140 = "stablehlo.constant"() <{value = dense<127> : tensor<i8>}> : () -> tensor<i8>
    %141 = "stablehlo.constant"() <{value = dense<-128> : tensor<i8>}> : () -> tensor<i8>
    %142 = "stablehlo.constant"() <{value = dense<127> : tensor<i8>}> : () -> tensor<i8>
    %143 = "func.call"(%140, %141, %142) <{callee = @clip}> : (tensor<i8>, tensor<i8>, tensor<i8>) -> tensor<i8>
    %144 = "stablehlo.convert"(%143) : (tensor<i8>) -> tensor<i64>
    %145 = "stablehlo.compare"(%arg43, %144) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %146 = "stablehlo.constant"() <{value = dense<-128> : tensor<i64>}> : () -> tensor<i64>
    %147 = "stablehlo.constant"() <{value = dense<127> : tensor<i64>}> : () -> tensor<i64>
    %148 = "func.call"(%arg42, %146, %147) <{callee = @clip_0}> : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %149 = "stablehlo.convert"(%148) : (tensor<i64>) -> tensor<i8>
    %150 = "stablehlo.constant"() <{value = dense<-128> : tensor<i64>}> : () -> tensor<i64>
    %151 = "stablehlo.constant"() <{value = dense<127> : tensor<i64>}> : () -> tensor<i64>
    %152 = "func.call"(%arg43, %150, %151) <{callee = @clip_0}> : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %153 = "stablehlo.convert"(%152) : (tensor<i64>) -> tensor<i8>
    %154 = "stablehlo.broadcast_in_dim"(%149) <{broadcast_dimensions = array<i64>}> : (tensor<i8>) -> tensor<1xi8>
    %155 = "stablehlo.broadcast_in_dim"(%153) <{broadcast_dimensions = array<i64>}> : (tensor<i8>) -> tensor<1xi8>
    %156 = "func.call"(%arg41) <{callee = @_threefry_split}> : (tensor<2xui32>) -> tensor<2x2xui32>
    %157 = "stablehlo.slice"(%156) <{limit_indices = array<i64: 1, 2>, start_indices = array<i64: 0, 0>, strides = array<i64: 1, 1>}> : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %158 = "stablehlo.reshape"(%157) : (tensor<1x2xui32>) -> tensor<2xui32>
    %159 = "stablehlo.slice"(%156) <{limit_indices = array<i64: 2, 2>, start_indices = array<i64: 1, 0>, strides = array<i64: 1, 1>}> : (tensor<2x2xui32>) -> tensor<1x2xui32>
    %160 = "stablehlo.reshape"(%159) : (tensor<1x2xui32>) -> tensor<2xui32>
    %161 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<8xui32>
    %162 = "stablehlo.slice"(%158) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %163 = "stablehlo.reshape"(%162) : (tensor<1xui32>) -> tensor<ui32>
    %164 = "stablehlo.slice"(%158) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %165 = "stablehlo.reshape"(%164) : (tensor<1xui32>) -> tensor<ui32>
    %166 = "stablehlo.slice"(%161) <{limit_indices = array<i64: 4>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<8xui32>) -> tensor<4xui32>
    %167 = "stablehlo.slice"(%161) <{limit_indices = array<i64: 8>, start_indices = array<i64: 4>, strides = array<i64: 1>}> : (tensor<8xui32>) -> tensor<4xui32>
    %168 = "stablehlo.constant"() <{value = dense<[13, 15, 26, 6]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %169 = "stablehlo.constant"() <{value = dense<[17, 29, 16, 24]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %170 = "stablehlo.xor"(%163, %165) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %171 = "stablehlo.constant"() <{value = dense<466688986> : tensor<ui32>}> : () -> tensor<ui32>
    %172 = "stablehlo.xor"(%170, %171) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %173 = "stablehlo.broadcast_in_dim"(%163) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %174 = "stablehlo.add"(%166, %173) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %175 = "stablehlo.broadcast_in_dim"(%165) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %176 = "stablehlo.add"(%167, %175) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %177 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %178 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %179:9 = "stablehlo.while"(%178, %177, %174, %176, %165, %172, %163, %168, %169) ({
    ^bb0(%arg71: tensor<i64>, %arg72: tensor<i64>, %arg73: tensor<4xui32>, %arg74: tensor<4xui32>, %arg75: tensor<ui32>, %arg76: tensor<ui32>, %arg77: tensor<ui32>, %arg78: tensor<4xui32>, %arg79: tensor<4xui32>):
      %267 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
      %268 = "stablehlo.compare"(%arg71, %267) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%268) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg62: tensor<i64>, %arg63: tensor<i64>, %arg64: tensor<4xui32>, %arg65: tensor<4xui32>, %arg66: tensor<ui32>, %arg67: tensor<ui32>, %arg68: tensor<ui32>, %arg69: tensor<4xui32>, %arg70: tensor<4xui32>):
      %264:8 = "func.call"(%arg63, %arg64, %arg65, %arg66, %arg67, %arg68, %arg69, %arg70) <{callee = @None_1}> : (tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %265 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %266 = "stablehlo.add"(%arg62, %265) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%266, %264#0, %264#1, %264#2, %264#3, %264#4, %264#5, %264#6, %264#7) : (tensor<i64>, tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
    }) : (tensor<i64>, tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
    %180 = "stablehlo.concatenate"(%179#2, %179#3) <{dimension = 0 : i64}> : (tensor<4xui32>, tensor<4xui32>) -> tensor<8xui32>
    %181 = "stablehlo.broadcast_in_dim"(%180) <{broadcast_dimensions = array<i64: 1>}> : (tensor<8xui32>) -> tensor<1x8xui32>
    %182 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4x1xui32>
    %183 = "stablehlo.constant"() <{value = dense<8> : tensor<ui32>}> : () -> tensor<ui32>
    %184 = "stablehlo.broadcast_in_dim"(%183) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4x1xui32>
    %185 = "stablehlo.multiply"(%184, %182) : (tensor<4x1xui32>, tensor<4x1xui32>) -> tensor<4x1xui32>
    %186 = "stablehlo.broadcast_in_dim"(%181) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x8xui32>) -> tensor<4x8xui32>
    %187 = "stablehlo.broadcast_in_dim"(%185) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<4x1xui32>) -> tensor<4x8xui32>
    %188 = "stablehlo.shift_right_logical"(%186, %187) : (tensor<4x8xui32>, tensor<4x8xui32>) -> tensor<4x8xui32>
    %189 = "stablehlo.constant"() <{value = dense<255> : tensor<ui32>}> : () -> tensor<ui32>
    %190 = "stablehlo.broadcast_in_dim"(%189) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4x8xui32>
    %191 = "stablehlo.and"(%190, %188) : (tensor<4x8xui32>, tensor<4x8xui32>) -> tensor<4x8xui32>
    %192 = "stablehlo.transpose"(%191) <{permutation = array<i64: 1, 0>}> : (tensor<4x8xui32>) -> tensor<8x4xui32>
    %193 = "stablehlo.reshape"(%192) : (tensor<8x4xui32>) -> tensor<32xui32>
    %194 = "stablehlo.convert"(%193) : (tensor<32xui32>) -> tensor<32xui8>
    %195 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<8xui32>
    %196 = "stablehlo.slice"(%160) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %197 = "stablehlo.reshape"(%196) : (tensor<1xui32>) -> tensor<ui32>
    %198 = "stablehlo.slice"(%160) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %199 = "stablehlo.reshape"(%198) : (tensor<1xui32>) -> tensor<ui32>
    %200 = "stablehlo.slice"(%195) <{limit_indices = array<i64: 4>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<8xui32>) -> tensor<4xui32>
    %201 = "stablehlo.slice"(%195) <{limit_indices = array<i64: 8>, start_indices = array<i64: 4>, strides = array<i64: 1>}> : (tensor<8xui32>) -> tensor<4xui32>
    %202 = "stablehlo.constant"() <{value = dense<[13, 15, 26, 6]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %203 = "stablehlo.constant"() <{value = dense<[17, 29, 16, 24]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %204 = "stablehlo.xor"(%197, %199) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %205 = "stablehlo.constant"() <{value = dense<466688986> : tensor<ui32>}> : () -> tensor<ui32>
    %206 = "stablehlo.xor"(%204, %205) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %207 = "stablehlo.broadcast_in_dim"(%197) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %208 = "stablehlo.add"(%200, %207) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %209 = "stablehlo.broadcast_in_dim"(%199) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %210 = "stablehlo.add"(%201, %209) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %211 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %212 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %213:9 = "stablehlo.while"(%212, %211, %208, %210, %199, %206, %197, %202, %203) ({
    ^bb0(%arg53: tensor<i64>, %arg54: tensor<i64>, %arg55: tensor<4xui32>, %arg56: tensor<4xui32>, %arg57: tensor<ui32>, %arg58: tensor<ui32>, %arg59: tensor<ui32>, %arg60: tensor<4xui32>, %arg61: tensor<4xui32>):
      %262 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
      %263 = "stablehlo.compare"(%arg53, %262) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%263) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg44: tensor<i64>, %arg45: tensor<i64>, %arg46: tensor<4xui32>, %arg47: tensor<4xui32>, %arg48: tensor<ui32>, %arg49: tensor<ui32>, %arg50: tensor<ui32>, %arg51: tensor<4xui32>, %arg52: tensor<4xui32>):
      %259:8 = "func.call"(%arg45, %arg46, %arg47, %arg48, %arg49, %arg50, %arg51, %arg52) <{callee = @None_1}> : (tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %260 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %261 = "stablehlo.add"(%arg44, %260) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%261, %259#0, %259#1, %259#2, %259#3, %259#4, %259#5, %259#6, %259#7) : (tensor<i64>, tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
    }) : (tensor<i64>, tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
    %214 = "stablehlo.concatenate"(%213#2, %213#3) <{dimension = 0 : i64}> : (tensor<4xui32>, tensor<4xui32>) -> tensor<8xui32>
    %215 = "stablehlo.broadcast_in_dim"(%214) <{broadcast_dimensions = array<i64: 1>}> : (tensor<8xui32>) -> tensor<1x8xui32>
    %216 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4x1xui32>
    %217 = "stablehlo.constant"() <{value = dense<8> : tensor<ui32>}> : () -> tensor<ui32>
    %218 = "stablehlo.broadcast_in_dim"(%217) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4x1xui32>
    %219 = "stablehlo.multiply"(%218, %216) : (tensor<4x1xui32>, tensor<4x1xui32>) -> tensor<4x1xui32>
    %220 = "stablehlo.broadcast_in_dim"(%215) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x8xui32>) -> tensor<4x8xui32>
    %221 = "stablehlo.broadcast_in_dim"(%219) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<4x1xui32>) -> tensor<4x8xui32>
    %222 = "stablehlo.shift_right_logical"(%220, %221) : (tensor<4x8xui32>, tensor<4x8xui32>) -> tensor<4x8xui32>
    %223 = "stablehlo.constant"() <{value = dense<255> : tensor<ui32>}> : () -> tensor<ui32>
    %224 = "stablehlo.broadcast_in_dim"(%223) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4x8xui32>
    %225 = "stablehlo.and"(%224, %222) : (tensor<4x8xui32>, tensor<4x8xui32>) -> tensor<4x8xui32>
    %226 = "stablehlo.transpose"(%225) <{permutation = array<i64: 1, 0>}> : (tensor<4x8xui32>) -> tensor<8x4xui32>
    %227 = "stablehlo.reshape"(%226) : (tensor<8x4xui32>) -> tensor<32xui32>
    %228 = "stablehlo.convert"(%227) : (tensor<32xui32>) -> tensor<32xui8>
    %229 = "stablehlo.subtract"(%155, %154) : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    %230 = "stablehlo.convert"(%229) : (tensor<1xi8>) -> tensor<1xui8>
    %231 = "stablehlo.compare"(%155, %154) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi1>
    %232 = "stablehlo.constant"() <{value = dense<1> : tensor<ui8>}> : () -> tensor<ui8>
    %233 = "stablehlo.broadcast_in_dim"(%232) <{broadcast_dimensions = array<i64>}> : (tensor<ui8>) -> tensor<1xui8>
    %234 = "stablehlo.select"(%231, %233, %230) : (tensor<1xi1>, tensor<1xui8>, tensor<1xui8>) -> tensor<1xui8>
    %235 = "stablehlo.compare"(%155, %154) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi1>
    %236 = "stablehlo.broadcast_in_dim"(%145) <{broadcast_dimensions = array<i64>}> : (tensor<i1>) -> tensor<1xi1>
    %237 = "stablehlo.and"(%236, %235) : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %238 = "stablehlo.constant"() <{value = dense<1> : tensor<ui8>}> : () -> tensor<ui8>
    %239 = "stablehlo.broadcast_in_dim"(%238) <{broadcast_dimensions = array<i64>}> : (tensor<ui8>) -> tensor<1xui8>
    %240 = "stablehlo.add"(%234, %239) : (tensor<1xui8>, tensor<1xui8>) -> tensor<1xui8>
    %241 = "stablehlo.select"(%237, %240, %234) : (tensor<1xi1>, tensor<1xui8>, tensor<1xui8>) -> tensor<1xui8>
    %242 = "stablehlo.constant"() <{value = dense<16> : tensor<ui8>}> : () -> tensor<ui8>
    %243 = "stablehlo.broadcast_in_dim"(%242) <{broadcast_dimensions = array<i64>}> : (tensor<ui8>) -> tensor<1xui8>
    %244 = "stablehlo.remainder"(%243, %241) : (tensor<1xui8>, tensor<1xui8>) -> tensor<1xui8>
    %245 = "stablehlo.multiply"(%244, %244) : (tensor<1xui8>, tensor<1xui8>) -> tensor<1xui8>
    %246 = "stablehlo.remainder"(%245, %241) : (tensor<1xui8>, tensor<1xui8>) -> tensor<1xui8>
    %247 = "stablehlo.broadcast_in_dim"(%241) <{broadcast_dimensions = array<i64: 0>}> : (tensor<1xui8>) -> tensor<32xui8>
    %248 = "stablehlo.remainder"(%194, %247) : (tensor<32xui8>, tensor<32xui8>) -> tensor<32xui8>
    %249 = "stablehlo.broadcast_in_dim"(%246) <{broadcast_dimensions = array<i64: 0>}> : (tensor<1xui8>) -> tensor<32xui8>
    %250 = "stablehlo.multiply"(%248, %249) : (tensor<32xui8>, tensor<32xui8>) -> tensor<32xui8>
    %251 = "stablehlo.broadcast_in_dim"(%241) <{broadcast_dimensions = array<i64: 0>}> : (tensor<1xui8>) -> tensor<32xui8>
    %252 = "stablehlo.remainder"(%228, %251) : (tensor<32xui8>, tensor<32xui8>) -> tensor<32xui8>
    %253 = "stablehlo.add"(%250, %252) : (tensor<32xui8>, tensor<32xui8>) -> tensor<32xui8>
    %254 = "stablehlo.broadcast_in_dim"(%241) <{broadcast_dimensions = array<i64: 0>}> : (tensor<1xui8>) -> tensor<32xui8>
    %255 = "stablehlo.remainder"(%253, %254) : (tensor<32xui8>, tensor<32xui8>) -> tensor<32xui8>
    %256 = "stablehlo.convert"(%255) : (tensor<32xui8>) -> tensor<32xi8>
    %257 = "stablehlo.broadcast_in_dim"(%154) <{broadcast_dimensions = array<i64: 0>}> : (tensor<1xi8>) -> tensor<32xi8>
    %258 = "stablehlo.add"(%257, %256) : (tensor<32xi8>, tensor<32xi8>) -> tensor<32xi8>
    "func.return"(%258) : (tensor<32xi8>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], function_type = (tensor<i8>, tensor<i8>, tensor<i8>) -> tensor<i8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "clip", sym_visibility = "private"}> ({
  ^bb0(%arg38: tensor<i8>, %arg39: tensor<i8>, %arg40: tensor<i8>):
    %138 = "stablehlo.maximum"(%arg39, %arg38) : (tensor<i8>, tensor<i8>) -> tensor<i8>
    %139 = "stablehlo.minimum"(%arg40, %138) : (tensor<i8>, tensor<i8>) -> tensor<i8>
    "func.return"(%139) : (tensor<i8>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], function_type = (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<i64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "clip_0", sym_visibility = "private"}> ({
  ^bb0(%arg35: tensor<i64>, %arg36: tensor<i64>, %arg37: tensor<i64>):
    %136 = "stablehlo.maximum"(%arg36, %arg35) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %137 = "stablehlo.minimum"(%arg37, %136) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    "func.return"(%137) : (tensor<i64>) -> ()
  }) : () -> ()
  "func.func"() <{arg_attrs = [{mhlo.layout_mode = "default"}], function_type = (tensor<2xui32>) -> tensor<2x2xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "_threefry_split", sym_visibility = "private"}> ({
  ^bb0(%arg16: tensor<2xui32>):
    %110 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<4xui32>
    %111 = "stablehlo.slice"(%arg16) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %112 = "stablehlo.reshape"(%111) : (tensor<1xui32>) -> tensor<ui32>
    %113 = "stablehlo.slice"(%arg16) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<2xui32>) -> tensor<1xui32>
    %114 = "stablehlo.reshape"(%113) : (tensor<1xui32>) -> tensor<ui32>
    %115 = "stablehlo.slice"(%110) <{limit_indices = array<i64: 2>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<2xui32>
    %116 = "stablehlo.slice"(%110) <{limit_indices = array<i64: 4>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<2xui32>
    %117 = "stablehlo.constant"() <{value = dense<[13, 15, 26, 6]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %118 = "stablehlo.constant"() <{value = dense<[17, 29, 16, 24]> : tensor<4xui32>}> : () -> tensor<4xui32>
    %119 = "stablehlo.xor"(%112, %114) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %120 = "stablehlo.constant"() <{value = dense<466688986> : tensor<ui32>}> : () -> tensor<ui32>
    %121 = "stablehlo.xor"(%119, %120) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %122 = "stablehlo.broadcast_in_dim"(%112) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %123 = "stablehlo.add"(%115, %122) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %124 = "stablehlo.broadcast_in_dim"(%114) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %125 = "stablehlo.add"(%116, %124) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %126 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %127 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %128:9 = "stablehlo.while"(%127, %126, %123, %125, %114, %121, %112, %117, %118) ({
    ^bb0(%arg26: tensor<i64>, %arg27: tensor<i64>, %arg28: tensor<2xui32>, %arg29: tensor<2xui32>, %arg30: tensor<ui32>, %arg31: tensor<ui32>, %arg32: tensor<ui32>, %arg33: tensor<4xui32>, %arg34: tensor<4xui32>):
      %134 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
      %135 = "stablehlo.compare"(%arg26, %134) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%135) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg17: tensor<i64>, %arg18: tensor<i64>, %arg19: tensor<2xui32>, %arg20: tensor<2xui32>, %arg21: tensor<ui32>, %arg22: tensor<ui32>, %arg23: tensor<ui32>, %arg24: tensor<4xui32>, %arg25: tensor<4xui32>):
      %131:8 = "func.call"(%arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25) <{callee = @None}> : (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
      %132 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
      %133 = "stablehlo.add"(%arg17, %132) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%133, %131#0, %131#1, %131#2, %131#3, %131#4, %131#5, %131#6, %131#7) : (tensor<i64>, tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
    }) : (tensor<i64>, tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>)
    %129 = "stablehlo.concatenate"(%128#2, %128#3) <{dimension = 0 : i64}> : (tensor<2xui32>, tensor<2xui32>) -> tensor<4xui32>
    %130 = "stablehlo.reshape"(%129) : (tensor<4xui32>) -> tensor<2x2xui32>
    "func.return"(%130) : (tensor<2x2xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>), sym_name = "None", sym_visibility = "private"}> ({
  ^bb0(%arg8: tensor<i64>, %arg9: tensor<2xui32>, %arg10: tensor<2xui32>, %arg11: tensor<ui32>, %arg12: tensor<ui32>, %arg13: tensor<ui32>, %arg14: tensor<4xui32>, %arg15: tensor<4xui32>):
    %55 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %56 = "stablehlo.add"(%arg8, %55) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %57 = "stablehlo.slice"(%arg14) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %58 = "stablehlo.reshape"(%57) : (tensor<1xui32>) -> tensor<ui32>
    %59 = "stablehlo.add"(%arg9, %arg10) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %60 = "stablehlo.broadcast_in_dim"(%58) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %61 = "stablehlo.shift_left"(%arg10, %60) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %62 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %63 = "stablehlo.subtract"(%62, %58) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %64 = "stablehlo.broadcast_in_dim"(%63) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %65 = "stablehlo.shift_right_logical"(%arg10, %64) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %66 = "stablehlo.or"(%61, %65) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %67 = "stablehlo.xor"(%59, %66) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %68 = "stablehlo.slice"(%arg14) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %69 = "stablehlo.reshape"(%68) : (tensor<1xui32>) -> tensor<ui32>
    %70 = "stablehlo.add"(%59, %67) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %71 = "stablehlo.broadcast_in_dim"(%69) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %72 = "stablehlo.shift_left"(%67, %71) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %73 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %74 = "stablehlo.subtract"(%73, %69) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %75 = "stablehlo.broadcast_in_dim"(%74) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %76 = "stablehlo.shift_right_logical"(%67, %75) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %77 = "stablehlo.or"(%72, %76) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %78 = "stablehlo.xor"(%70, %77) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %79 = "stablehlo.slice"(%arg14) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %80 = "stablehlo.reshape"(%79) : (tensor<1xui32>) -> tensor<ui32>
    %81 = "stablehlo.add"(%70, %78) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %82 = "stablehlo.broadcast_in_dim"(%80) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %83 = "stablehlo.shift_left"(%78, %82) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %84 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %85 = "stablehlo.subtract"(%84, %80) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %86 = "stablehlo.broadcast_in_dim"(%85) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %87 = "stablehlo.shift_right_logical"(%78, %86) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %88 = "stablehlo.or"(%83, %87) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %89 = "stablehlo.xor"(%81, %88) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %90 = "stablehlo.slice"(%arg14) <{limit_indices = array<i64: 4>, start_indices = array<i64: 3>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %91 = "stablehlo.reshape"(%90) : (tensor<1xui32>) -> tensor<ui32>
    %92 = "stablehlo.add"(%81, %89) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %93 = "stablehlo.broadcast_in_dim"(%91) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %94 = "stablehlo.shift_left"(%89, %93) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %95 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %96 = "stablehlo.subtract"(%95, %91) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %97 = "stablehlo.broadcast_in_dim"(%96) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %98 = "stablehlo.shift_right_logical"(%89, %97) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %99 = "stablehlo.or"(%94, %98) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %100 = "stablehlo.xor"(%92, %99) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %101 = "stablehlo.broadcast_in_dim"(%arg11) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %102 = "stablehlo.add"(%92, %101) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %103 = "stablehlo.broadcast_in_dim"(%arg12) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %104 = "stablehlo.add"(%100, %103) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    %105 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %106 = "stablehlo.add"(%arg8, %105) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %107 = "stablehlo.convert"(%106) : (tensor<i64>) -> tensor<ui32>
    %108 = "stablehlo.broadcast_in_dim"(%107) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<2xui32>
    %109 = "stablehlo.add"(%104, %108) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    "func.return"(%56, %102, %109, %arg12, %arg13, %arg11, %arg15, %arg14) : (tensor<i64>, tensor<2xui32>, tensor<2xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> (tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>), sym_name = "None_1", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<4xui32>, %arg2: tensor<4xui32>, %arg3: tensor<ui32>, %arg4: tensor<ui32>, %arg5: tensor<ui32>, %arg6: tensor<4xui32>, %arg7: tensor<4xui32>):
    %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %1 = "stablehlo.add"(%arg0, %0) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %2 = "stablehlo.slice"(%arg6) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %3 = "stablehlo.reshape"(%2) : (tensor<1xui32>) -> tensor<ui32>
    %4 = "stablehlo.add"(%arg1, %arg2) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %5 = "stablehlo.broadcast_in_dim"(%3) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %6 = "stablehlo.shift_left"(%arg2, %5) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %7 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %8 = "stablehlo.subtract"(%7, %3) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %9 = "stablehlo.broadcast_in_dim"(%8) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %10 = "stablehlo.shift_right_logical"(%arg2, %9) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %11 = "stablehlo.or"(%6, %10) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %12 = "stablehlo.xor"(%4, %11) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %13 = "stablehlo.slice"(%arg6) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %14 = "stablehlo.reshape"(%13) : (tensor<1xui32>) -> tensor<ui32>
    %15 = "stablehlo.add"(%4, %12) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %16 = "stablehlo.broadcast_in_dim"(%14) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %17 = "stablehlo.shift_left"(%12, %16) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %18 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %19 = "stablehlo.subtract"(%18, %14) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %20 = "stablehlo.broadcast_in_dim"(%19) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %21 = "stablehlo.shift_right_logical"(%12, %20) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %22 = "stablehlo.or"(%17, %21) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %23 = "stablehlo.xor"(%15, %22) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %24 = "stablehlo.slice"(%arg6) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %25 = "stablehlo.reshape"(%24) : (tensor<1xui32>) -> tensor<ui32>
    %26 = "stablehlo.add"(%15, %23) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %27 = "stablehlo.broadcast_in_dim"(%25) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %28 = "stablehlo.shift_left"(%23, %27) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %29 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %30 = "stablehlo.subtract"(%29, %25) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %31 = "stablehlo.broadcast_in_dim"(%30) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %32 = "stablehlo.shift_right_logical"(%23, %31) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %33 = "stablehlo.or"(%28, %32) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %34 = "stablehlo.xor"(%26, %33) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %35 = "stablehlo.slice"(%arg6) <{limit_indices = array<i64: 4>, start_indices = array<i64: 3>, strides = array<i64: 1>}> : (tensor<4xui32>) -> tensor<1xui32>
    %36 = "stablehlo.reshape"(%35) : (tensor<1xui32>) -> tensor<ui32>
    %37 = "stablehlo.add"(%26, %34) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %38 = "stablehlo.broadcast_in_dim"(%36) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %39 = "stablehlo.shift_left"(%34, %38) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %40 = "stablehlo.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
    %41 = "stablehlo.subtract"(%40, %36) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
    %42 = "stablehlo.broadcast_in_dim"(%41) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %43 = "stablehlo.shift_right_logical"(%34, %42) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %44 = "stablehlo.or"(%39, %43) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %45 = "stablehlo.xor"(%37, %44) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %46 = "stablehlo.broadcast_in_dim"(%arg3) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %47 = "stablehlo.add"(%37, %46) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %48 = "stablehlo.broadcast_in_dim"(%arg4) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %49 = "stablehlo.add"(%45, %48) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    %50 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %51 = "stablehlo.add"(%arg0, %50) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %52 = "stablehlo.convert"(%51) : (tensor<i64>) -> tensor<ui32>
    %53 = "stablehlo.broadcast_in_dim"(%52) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<4xui32>
    %54 = "stablehlo.add"(%49, %53) : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xui32>
    "func.return"(%1, %47, %54, %arg4, %arg5, %arg3, %arg7, %arg6) : (tensor<i64>, tensor<4xui32>, tensor<4xui32>, tensor<ui32>, tensor<ui32>, tensor<ui32>, tensor<4xui32>, tensor<4xui32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

